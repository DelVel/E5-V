import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from datasets import load_dataset
from peft import PeftModel
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils import data
from tqdm import tqdm
from transformers import (
    BitsAndBytesConfig,
    LlavaNextProcessor,
)

from ft_llm import LlavaNextCustom


def recall_at_k(scores: torch.Tensor, pos_pairs: torch.Tensor, k):
    top_idx = scores.topk(k, dim=1).indices
    retrieved_pos = pos_pairs.gather(1, top_idx)
    total_retrieved = retrieved_pos.sum(dim=1)
    total_pos = pos_pairs.sum(dim=1)
    return (total_retrieved / total_pos).mean()


class FIQCollator:
    def __init__(self, processor, fiq_data_name, mode):
        assert mode in ["query", "target"]
        self._processor = processor
        self._fiq_data_name = fiq_data_name
        self._mode = mode

    def __call__(self, data_):
        if self._mode == "query":
            texts = [x["caption"] for x in data_]
            images = [x["candidate"] for x in data_]
            id_ = [x["target_id"] for x in data_]
        else:
            texts = [None] * len(data_)
            images = [x["img"] for x in data_]
            id_ = [x["id"] for x in data_]

        templated = self._processor.apply_chat_template(
            [self._get_templated_prompt(txt) for txt in texts],
            add_generation_prompt=True,
        )

        return (
            self._processor(
                pad_to_multiple_of=8,
                padding=True,
                padding_side="left",
                return_tensors="pt",
                text=templated,
                images=images,
            ),
            np.array(id_),
        )

    def _get_templated_prompt(self, txt):
        base_text = (
            f"Describe this {self._fiq_data_name} in one word based on its style:"
        )
        if self._mode == "query":
            text = (
                f"Change the style of this {self._fiq_data_name} to {txt}\n" + base_text
            )
        else:
            text = base_text

        user_msg = {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text},
            ],
        }
        return [user_msg]


def prepare_cir_query_dataloader(accelerator, transform, style="shirt"):
    d0fm = (
        load_dataset("royokong/fashioniq_val")["val"]
        .filter(lambda x: map(lambda y: y == style, x["category"]), batched=True)
        .map(
            lambda x: {
                "caption": list(
                    map(lambda y: ", ".join(cc.strip(".?, ") for cc in y), x["caption"])
                )
            },
            batched=True,
        )
    )
    d0fl = get_dataloader(d0fm, FIQCollator(transform, style, "query"))
    d0f1m = accelerator.prepare(d0fl)
    return d0f1m


def prepare_cir_target_dataloader(accelerator, transform, style="shirt"):
    d1fm = (
        load_dataset("royokong/fashioniq_val_imgs")["val"]
        .filter(lambda x: map(lambda y: y == style, x["category"]), batched=True)
        .remove_columns(["category", "split"])
    )
    d1fl = get_dataloader(d1fm, FIQCollator(transform, style, "target"))
    d1f1m = accelerator.prepare(d1fl)
    return d1f1m


def get_dataloader(dataset, collate_fn):
    return data.DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=4,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
    )


def init_model(lora_path, accelerator: Accelerator):
    model_name = "llava-hf/llama3-llava-next-8b-hf"

    model_dtype = torch.bfloat16
    device = accelerator.device

    with torch.cuda.device(device):
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
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()

    model = accelerator.prepare(model)
    model.eval()
    return model


def extract_query_embeds(accelerator: Accelerator, dataloader, model):
    results = []
    labels = []
    if accelerator.is_main_process:
        dataloader = tqdm(dataloader)
    for batch in dataloader:
        with torch.inference_mode():
            result = model(
                **batch[0], output_hidden_states=True, return_dict=True
            ).hidden_states[-1][:, -1, :]
        results.extend(accelerator.gather_for_metrics(result))
        labels.extend(accelerator.gather_for_metrics(batch[1]))
    results = torch.stack(results)
    results = F.normalize(results, p=2, dim=1)
    labels = np.stack(labels)
    return results, labels


def extract_target_embeds(accelerator: Accelerator, dataloader, model):
    return extract_query_embeds(accelerator, dataloader, model)


@record
def main():
    accelerator = Accelerator()

    model_name = "llava-hf/llama3-llava-next-8b-hf"
    model = init_model("e5v-8b", accelerator)
    transform = LlavaNextProcessor.from_pretrained(model_name)

    q_dataloader = prepare_cir_query_dataloader(accelerator, transform)
    t_dataloader = prepare_cir_target_dataloader(accelerator, transform)

    q_emb, q_id = extract_query_embeds(accelerator, q_dataloader, model)
    t_emb, t_id = extract_target_embeds(accelerator, t_dataloader, model)

    if accelerator.is_main_process:
        score = torch.einsum("ij,kj->ik", q_emb, t_emb)
        pos_pairs = q_id[:, None] == t_id[None, :]
        pos_pairs = torch.from_numpy(pos_pairs).to(device=accelerator.device)

        r_a_10 = recall_at_k(score, pos_pairs, 10)
        r_a_50 = recall_at_k(score, pos_pairs, 50)
        print(f"Recall@10: {r_a_10}, Recall@50: {r_a_50}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
