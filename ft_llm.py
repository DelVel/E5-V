from dataclasses import dataclass
from pathlib import Path

import accelerate
import datasets
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from jsonargparse import CLI
from peft.tuners.lora.config import LoraConfig
from peft.mapping import get_peft_model
from peft.utils.other import prepare_model_for_kbit_training
from torch.distributed.elastic.multiprocessing import errors
from transformers import (
    BitsAndBytesConfig,
    LlavaConfig,
    LlavaProcessor,
    set_seed,
)
from transformers.trainer_utils import (
    set_seed,
)

from src.data import (
    get_cc3m_dataset_with_img_embed,
    get_fiq_image_dataset,
    get_fiq_text_dataset,
    prompt_image_text,
)
from src.model import LlavaCustom, LlavaCustomProcessor
from src.trainer.trainer import SentembTrainer


def relaxed_contrastive_loss(t_emb, s_emb, sigma=1, delta=1):
    with torch.no_grad():
        s_emb = F.normalize(s_emb, p=2, dim=1)
        S_dist = torch.cdist(s_emb, s_emb)
        P = torch.exp(-S_dist.pow(2) / sigma)

    T_dist = torch.cdist(t_emb, t_emb)
    T_dist = T_dist / T_dist.mean(1)

    pull_losses = P * T_dist.pow(2)
    push_losses = (1 - P) * (delta - T_dist).clamp(0).pow(2)

    loss = (pull_losses.sum() + push_losses.sum()) / len(t_emb)
    return loss



class DataCollator:
    def __init__(self, processor):
        self._processor: LlavaProcessor = processor

    def __call__(self, data_):
        images = [x['jpg'] for x in data_]
        images_templated = self._processor.apply_chat_template(
            [prompt_image_text("Summary above image in one word:") for _ in images],
            add_generation_prompt=True,
        )
        images_processed = self._processor(
            images=images,
            text=images_templated,
            pad_to_multiple_of=8,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        )
        images_processed["label"] = [x['clip_img_embed'] for x in data_]
        return images_processed



@dataclass
class LoraParams:
    r: int
    alpha: int
    dropout: float
    target_modules: list[str]


def get_processor(model_name):
    processor: LlavaCustomProcessor = LlavaCustomProcessor.from_pretrained(model_name)
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


@errors.record
def main(
    output_dir: str,
    lora: LoraParams,
    run_name: str | None = None,
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

        args = transformers.TrainingArguments(
            bf16=bf16,
            dataloader_num_workers=0,
            ddp_find_unused_parameters=False,
            deepspeed="ds_config.json",
            eval_strategy="steps",
            report_to="tensorboard",
            eval_steps=100,
            eval_on_start=False,
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

        train_data = get_cc3m_dataset_with_img_embed()
        data_collator = DataCollator(processor)
        trainer = SentembTrainer(
            args=args,
            data_collator=data_collator,
            model=model,
            processing_class=processor,
            train_dataset=train_data,
            eval_dataset={
                "fiq_dress_query": get_fiq_text_dataset(processor, "dress"),
                "fiq_dress_target": get_fiq_image_dataset(processor, "dress"),
            },
        )
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    finally:
        trainer.save_model()
        accelerator.end_training()

if __name__ == "__main__":
    CLI(main)
