import numpy as np
import peft
import torch
import transformers
from peft import LoraConfig
from torch.distributed.elastic.multiprocessing import errors
from torch.nn import functional as F
from transformers import (
    AutoProcessor,
    CLIPProcessor,
    CLIPTextModelWithProjection,
    LlavaForConditionalGeneration,
)
from trl import GRPOConfig

from retrieval import calculate_pos_pairs
from src.datasets.fashion_iq import get_fiq_image_dataset, get_fiq_label_dataset
from src.trainer.custom_grpo_trainer import CustomGRPOTrainer


class Grader:
    def __init__(self):
        self.it_id = np.array(
            get_fiq_image_dataset("train", ["dress", "shirt", "toptee"])["id"]
        )
        clip_model = "openai/clip-vit-large-patch14"
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model, use_fast=True)
        self.clip_text_model = CLIPTextModelWithProjection.from_pretrained(
            clip_model
        ).to("cuda")
        self.clip_img_embed = torch.load(
            "fiq_clip_emb.pt", map_location="cuda", weights_only=True
        )

    def __call__(self, x_input, tgt_ids):
        processed = self.clip_processor(
            x_input,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.clip_text_model.config.max_position_embeddings,
        ).to("cuda")
        tid = processed["input_ids"]
        attn = processed["attention_mask"]
        with torch.inference_mode():
            text_features = self.clip_text_model(
                input_ids=tid, attention_mask=attn
            ).text_embeds
        text_features = F.normalize(text_features, p=2, dim=-1)
        tgt_np_id = np.array(tgt_ids)

        scores = text_features @ self.clip_img_embed.T
        positive_pairs = calculate_pos_pairs(tgt_np_id, self.it_id)

        # reciprocal rank
        argsort_idx = scores.argsort(descending=True, dim=1)
        rank = positive_pairs.gather(1, argsort_idx).int().argmax(dim=1) + 1

        return (100 / rank).tolist()


_grader = Grader()


def map_func(x):
    return {
        "prompt": [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": f"In one sentence, describe this image modified as {' and '.join(xx)}.",
                        },
                    ],
                }
            ]
            for xx in x["tq"]
        ]
    }


def retrieval_score(completions, **kwargs):
    all_completions = [x[0]["content"] for x in completions]
    all_it_id = kwargs["it_id"]
    grades = _grader(all_completions, all_it_id)
    max_idx = max(range(len(grades)), key=grades.__getitem__)
    print(
        f"{all_it_id[max_idx]} ({grades[max_idx]:6.2f}): {all_completions[max_idx]}"
    )
    return grades


@errors.record
def main():
    transformers.set_seed(42)

    rank = torch.distributed.get_rank()
    args = GRPOConfig(
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        learning_rate=1e-4,
        logging_steps=1,
        max_completion_length=77,
        max_prompt_length=None,
        num_generations=8,
        num_train_epochs=100,
        output_dir="runs/grpo-multi-vllm",
        per_device_train_batch_size=32,
        report_to="tensorboard",
        save_steps=100,
        save_strategy="steps",
        save_total_limit=3,
        use_vllm=True,
        vllm_gpu_memory_utilization=0.80,
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "down_proj",
            "up_proj",
        ],
        task_type="CAUSAL_LM",
    )

    model_dtype = torch.bfloat16
    model_name = "/home/taegyupark/.cache/huggingface/hub/models--xtuner--llava-phi-3-mini-hf/snapshots/218fa56e23d2b894dd13f2c4ecf4b90843b12b39"
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
    )
    model.enable_input_require_grads()
    model = peft.peft_model.PeftModel.from_pretrained(
        model, "runs/grpo5/checkpoint-200", is_trainable=True, torch_device=f"cuda:{rank}"
    )
    model.print_trainable_parameters()

    processor = AutoProcessor.from_pretrained(
        model_name, padding_side="left", use_fast=True
    )
    processor.pad_token_id = processor.tokenizer.eos_token_id

    dataset = (
        get_fiq_label_dataset("train", ["dress", "shirt", "toptee"])
        .remove_columns("iq_id")
        .map(
            map_func,
            batched=True,
            remove_columns=["tq", "it"],
        )
    )

    trainer = CustomGRPOTrainer(
        args=args,
        model=model,
        processing_class=processor,
        reward_funcs=[retrieval_score],
        train_dataset=dataset,
    )

    trainer.train(resume_from_checkpoint=False)


if __name__ == "__main__":
    main()
