from dataclasses import dataclass
from pathlib import Path

import accelerate
from torch.distributed.elastic.multiprocessing import errors
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
    model_dtype,
):
    model = LlavaCustom.from_pretrained(
        model_name,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
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
    return model


def get_dataset():
    cc3m = load_dataset("pixparse/cc3m-wds", split="train")
    cc3m = cc3m.shuffle()
    cc3m = cc3m.remove_columns(["__key__", "__url__"])
    return cc3m


@errors.record
def main(
    run_name: str,
    output_dir: str,
    lora: LoraParams,
    # training hyperparams
    per_device_train_batch_size: int = 64,
    gradient_accumulation_steps: int = 1,
    num_epochs: int = 1,
    learning_rate: float = 5e-4,
    bf16: bool = True,
    # trainer parameters
    resume_from_checkpoint: bool = False,
):
    accelerator = accelerate.Accelerator()
    try:
        set_seed(42)

        if not accelerator.is_main_process:
            transformers.utils.logging.disable_progress_bar()
            datasets.disable_progress_bars()
        else:
            print("Progress bars are disabled in non-main processes.")

        output_dir: Path = Path(output_dir) / run_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dir = str(output_dir)
        args = transformers.TrainingArguments(
            bf16=bf16,
            dataloader_num_workers=4,
            ddp_find_unused_parameters=False,
            deepspeed="ds_config.json",
            fp16=not bf16,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": True},
            learning_rate=learning_rate,
            logging_steps=1,
            num_train_epochs=num_epochs,
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            remove_unused_columns=False,
            run_name=run_name,
            save_steps=100,
            save_strategy="steps",
            save_total_limit=3,
            warmup_steps=100,
        )

        model_name = "xtuner/llava-phi-3-mini-hf"
        processor = get_processor(model_name)
        model = get_model(
            model_name,
            lora,
            torch.bfloat16 if bf16 else torch.float16,
        )
        if accelerator.is_main_process:
            model.print_trainable_parameters()
        
        train_data = get_dataset()
        data_collator = DataCollator(processor)
        trainer = SentembTrainer(
            args=args,
            data_collator=data_collator,
            model=model,
            processing_class=processor,
            train_dataset=train_data,
        )
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        model.save_pretrained(output_dir, is_main_process=accelerator.is_main_process)
    finally:
        accelerator.end_training()


if __name__ == "__main__":
    CLI(main)
