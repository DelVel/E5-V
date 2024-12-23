import contextlib
import os

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
    PreTrainedTokenizerFast,
    Trainer,
    set_seed,
)


class LlavaNextCustom(LlavaNextForConditionalGeneration):
    """
    A custom model that allows inputs without pixel values.
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


class DataTransform:
    def __init__(self, tokenizer):
        self._tokenizer: PreTrainedTokenizerFast = tokenizer

    def __call__(self, string):
        return {
            k: self._tokenizer.batch_decode(
                self._tokenizer(
                    v,
                    truncation=True,
                    max_length=32,
                    add_special_tokens=False,
                )["input_ids"]
            )
            for k, v in string.items()
        }


class DataCollator:
    _keys = ("sent0", "sent1", "hard_neg")

    def __init__(self, processor):
        self._processor: LlavaNextProcessor = processor

    def __call__(self, data_):
        flattened_data = [x[k] for k in self._keys for x in data_]
        templated = self._processor.apply_chat_template(
            [self._get_templated_prompt(x) for x in flattened_data],
            add_generation_prompt=True,
        )
        return self._processor(
            pad_to_multiple_of=8,
            padding=True,
            padding_side="left",
            return_tensors="pt",
            text=templated,
        )

    def _get_templated_prompt(self, x):
        text_content = {
            "type": "text",
            "text": f"{x}\nSummary above sentence in one word:",
        }
        user_msg = {"role": "user", "content": [text_content]}
        return [user_msg]


class SentembTrainer(Trainer):
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        pooler_output = model(
            output_hidden_states=True, return_dict=True, **inputs
        ).hidden_states[-1][:, -1, :]

        batch_size = pooler_output.size(0) // 3
        assert batch_size * 3 == pooler_output.size(0)

        z1 = pooler_output[:batch_size]
        z2 = pooler_output[batch_size : 2 * batch_size]
        z3 = pooler_output[2 * batch_size :]

        if dist.is_initialized():
            z1 = all_gather_with_grad(z1.contiguous())
            z2 = all_gather_with_grad(z2.contiguous())
            z3 = all_gather_with_grad(z3.contiguous())

        query = z1.unsqueeze_(1)
        target = torch.cat([z2.unsqueeze_(0), z3.unsqueeze_(0)], 1)
        cos_sim = F.cosine_similarity(query, target, dim=-1) / 0.05

        labels = torch.arange(cos_sim.size(0), dtype=torch.long, device=cos_sim.device)

        loss = F.cross_entropy(cos_sim, labels)

        return (loss, z1, z2, z3) if return_outputs else loss


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


def get_data(data_path, world_size, processor):
    data = load_dataset("csv", data_files=data_path)
    data_transform = DataTransform(processor.tokenizer)
    train_data = (
        data["train"]
        .shuffle()
        .map(data_transform, num_proc=os.cpu_count() // world_size, batched=True)
    )

    return train_data


def train(
    # model/data params
    data_path: str = "data/nli_for_simcse.csv",
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
    train_data = get_data(data_path, world_size, processor)
    data_collator = DataCollator(processor)

    with pgroup_context, torch.cuda.device(device):
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
            model=model,
            train_dataset=train_data,
            data_collator=data_collator,
            processing_class=processor,
            args=transformers.TrainingArguments(
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
                save_total_limit=100,
                warmup_steps=100,
            ),
        )
        trainer.train()
        model.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
