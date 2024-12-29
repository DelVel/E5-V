import contextlib
import os

import datasets
from einops import rearrange
import fire
import torch
import torch.distributed as dist
import torch.nn.functional as F
import transformers
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    BitsAndBytesConfig,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
    Trainer,
    set_seed,
)

from data import prompt_image_text, prompt_text


class LlavaNextCustom(LlavaNextForConditionalGeneration):
    """
    A custom model that allows both image and text inputs to be processed
    """

    def forward(self, *args, **kwargs):
        pixel_values = kwargs.get("pixel_values", None)
        if pixel_values is not None:
            return super().forward(*args, **kwargs)
        return self.language_model.forward(*args, **kwargs)


class GatherLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        output = [
            torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]


def all_gather_with_grad(tensors):
    world_size = torch.distributed.get_world_size()
    if world_size == 1:
        return tensors
    return torch.cat(GatherLayer.apply(tensors))


class PgroupContext:
    def __init__(self, device_id):
        self._device_id = device_id

    def __enter__(self):
        torch.distributed.init_process_group("nccl", device_id=self._device_id)

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.distributed.destroy_process_group()


class DataCollator:
    def __init__(self, processor):
        self._processor: LlavaNextProcessor = processor

    def __call__(self, data_):
        text = [x["txt"] for x in data_]
        text = self._processor.batch_decode(
            self._processor(
                text=text,
                truncation=True,
                max_length=32,
                add_special_tokens=False,
            )["input_ids"]
        )

        text_templated = self._processor.apply_chat_template(
            [prompt_text(f"{x}\nSummary above sentence in one word:") for x in text],
            add_generation_prompt=True,
        )

        images = [x["jpg"] for x in data_]

        images_templated = self._processor.apply_chat_template(
            [prompt_image_text("Summary above image in one word:") for _ in images],
            add_generation_prompt=True,
        )
        images_processed = self._processor(
            images=images,
            text=images_templated + text_templated,
            pad_to_multiple_of=8,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        )
        return images_processed


class SentembTrainer(Trainer):
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        outputs = model(
            **inputs, return_dict=True, output_hidden_states=True
        ).hidden_states[-1][:, -1, :]
        outputs = rearrange(outputs, "(m b) d -> m b d", m=2)
        img_outputs, txt_outputs = outputs[0], outputs[1]

        if dist.is_initialized():
            img_outputs = all_gather_with_grad(img_outputs.contiguous())
            txt_outputs = all_gather_with_grad(txt_outputs.contiguous())

        query = img_outputs.unsqueeze_(1)
        target = txt_outputs.unsqueeze_(0)
        cos_sim = F.cosine_similarity(query, target, dim=-1) / 0.05

        labels = torch.arange(cos_sim.size(0), dtype=torch.long, device=cos_sim.device)

        loss = (
            F.cross_entropy(cos_sim, labels) + F.cross_entropy(cos_sim.t(), labels)
        ) / 2

        return (loss, txt_outputs, img_outputs) if return_outputs else loss


def get_model(
    model_name,
    lora_r,
    lora_alpha,
    lora_dropout,
    lora_target_modules,
    grad_checkpoint,
    model_dtype,
    device,
):
    model = LlavaNextCustom.from_pretrained(
        model_name,
        torch_dtype=model_dtype,
        device_map=device,
        attn_implementation="flash_attention_2",
        quantization_config=BitsAndBytesConfig(
            bnb_4bit_compute_dtype=model_dtype,
            bnb_4bit_quant_storage=model_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            load_in_4bit=True,
        ),
    )

    if grad_checkpoint:
        model.enable_input_require_grads()

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            exclude_modules="^(?!language_model).*$",
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()
    return model


def get_data():
    cc3m = load_dataset("pixparse/cc3m-wds", split="train")
    cc3m = cc3m.shuffle()
    cc3m = cc3m.remove_columns(["__key__", "__url__"])
    return cc3m


def train(
    # model/data params
    output_dir: str = "./lora-alpaca",
    # training hyperparams
    batch_size: int = 256,
    micro_batch_size: int = 64,
    num_epochs: int = 1,
    learning_rate: float = 5e-4,
    # lora hyperparams
    lora_r: int = 64,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: list[str] = None,
    # llm hyperparams
    save_steps: int = 100,
    seed: int = 42,
    deepspeed: str = None,
    logging_steps: int = 10,
    grad_checkpoint: bool = True,
    bf16: bool = True,
    # ddp vars
    local_rank: str = None,
):
    set_seed(seed)

    if local_rank is not None and local_rank != 0:
        transformers.utils.logging.disable_progress_bar()
        datasets.disable_progress_bars()
        print(f"Disabling progress bars for rank {local_rank}")

    fp16 = True if not bf16 else False
    model_dtype = torch.bfloat16 if bf16 else torch.float16
    gradient_accumulation_steps = batch_size // micro_batch_size
    device = torch.device("cuda")
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    pgroup_context = contextlib.nullcontext()

    if ddp:
        assert local_rank is not None and isinstance(local_rank, int)
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
        device = torch.device("cuda", local_rank)
        pgroup_context = PgroupContext(device)

    model_name = "llava-hf/llama3-llava-next-8b-hf"
    processor: LlavaNextProcessor = LlavaNextProcessor.from_pretrained(model_name)
    train_data = get_data()
    data_collator = DataCollator(processor)

    with pgroup_context, torch.cuda.device(device):
        args = transformers.TrainingArguments(
            bf16=bf16,
            ddp_find_unused_parameters=False if ddp else None,
            deepspeed=deepspeed,
            fp16=fp16,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=grad_checkpoint,
            gradient_checkpointing_kwargs={"use_reentrant": True},
            learning_rate=learning_rate,
            logging_steps=logging_steps,
            num_train_epochs=num_epochs,
            output_dir=output_dir,
            per_device_train_batch_size=micro_batch_size,
            remove_unused_columns=False,
            run_name=output_dir,
            save_steps=save_steps,
            save_strategy="steps",
            save_total_limit=3,
            warmup_steps=100,
        )
        model = get_model(
            model_name,
            lora_r,
            lora_alpha,
            lora_dropout,
            lora_target_modules,
            grad_checkpoint,
            model_dtype,
            device,
        )
        trainer = SentembTrainer(
            args=args,
            data_collator=data_collator,
            model=model,
            processing_class=processor,
            train_dataset=train_data,
        )
        trainer.train()
        model.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
