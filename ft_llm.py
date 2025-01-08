import contextlib
import os
from dataclasses import dataclass

import datasets
import torch
import torch.distributed as dist
import torch.nn.functional as F
import transformers
from datasets import load_dataset
from einops import rearrange
from jsonargparse import CLI
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    BitsAndBytesConfig,
    LlavaConfig,
    LlavaForConditionalGeneration,
    LlavaProcessor,
    Trainer,
    set_seed,
)

from data import prompt_image_text, prompt_text


class LlavaCustom(LlavaForConditionalGeneration):
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


@contextlib.contextmanager
def pgroup_context(device_id):
    torch.distributed.init_process_group("nccl", device_id=device_id)
    yield
    torch.distributed.destroy_process_group()


class DataCollator:
    def __init__(self, processor):
        self._processor: LlavaProcessor = processor

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
            **inputs, return_dict=True, output_hidden_states=True, use_cache=False
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


@dataclass
class LoraParams:
    r: int
    alpha: int
    dropout: float
    target_modules: list[str]


def get_processor(model_name):
    processor = LlavaProcessor.from_pretrained(model_name)
    processor.chat_template = "{% for message in messages %}{{ '<|' + message['role'] + '|>\n'}}{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}{{ '<image>' }}{% endfor %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{{ '\n' + content['text'] + '<|end|>\n' }}{% endfor %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}"
    model_cfg = LlavaConfig.from_pretrained(model_name)
    processor.patch_size = model_cfg.vision_config.patch_size
    processor.vision_feature_select_strategy = model_cfg.vision_feature_select_strategy
    return processor


def get_model(
    model_name,
    lora_params: LoraParams,
    grad_checkpoint,
    model_dtype,
    device,
):
    model = LlavaCustom.from_pretrained(
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

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora_params.r,
            lora_alpha=lora_params.alpha,
            lora_dropout=lora_params.dropout,
            target_modules=lora_params.target_modules,
            exclude_modules="^(?!language_model).*$",
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()
    return model


def get_dataset():
    cc3m = load_dataset("pixparse/cc3m-wds", split="train")
    cc3m = cc3m.shuffle()
    cc3m = cc3m.remove_columns(["__key__", "__url__"])
    return cc3m


def train(
    output_dir: str,
    lora: LoraParams,
    # training hyperparams
    per_device_train_batch_size: int = 64,
    gradient_accumulation_steps: int = 1,
    num_epochs: int = 1,
    learning_rate: float = 5e-4,
    # llm hyperparams
    save_steps: int = 100,
    seed: int = 42,
    deepspeed: str = None,
    grad_checkpoint: bool = True,
    bf16: bool = True,
    # ddp vars
    local_rank: int = None,
):
    set_seed(seed)

    if local_rank is not None and local_rank != 0:
        transformers.utils.logging.disable_progress_bar()
        datasets.disable_progress_bars()
        print(f"Disabling progress bars for rank {local_rank}")

    fp16 = True if not bf16 else False
    model_dtype = torch.bfloat16 if bf16 else torch.float16
    device = torch.device("cuda")
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    nccl_ctx = contextlib.nullcontext()

    if ddp:
        assert local_rank is not None and isinstance(
            local_rank, int
        ), f"Invalid rank {local_rank}:{type(local_rank)}"
        device = torch.device("cuda", local_rank)
        nccl_ctx = pgroup_context(device)

    model_name = "xtuner/llava-phi-3-mini-hf"
    processor = get_processor(model_name)
    train_data = get_dataset()
    data_collator = DataCollator(processor)

    with nccl_ctx, torch.cuda.device(device):
        args = transformers.TrainingArguments(
            bf16=bf16,
            ddp_find_unused_parameters=False if ddp else None,
            deepspeed=deepspeed,
            fp16=fp16,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=grad_checkpoint,
            gradient_checkpointing_kwargs={"use_reentrant": True},
            learning_rate=learning_rate,
            logging_steps=1,
            num_train_epochs=num_epochs,
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            remove_unused_columns=False,
            run_name=output_dir,
            save_steps=save_steps,
            save_strategy="steps",
            save_total_limit=3,
            warmup_steps=100,
        )
        model = get_model(
            model_name,
            lora,
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
    CLI(train)
