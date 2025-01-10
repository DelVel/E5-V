import datetime
import os
from dataclasses import dataclass
from os import cpu_count
from pathlib import Path

import datasets
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from einops import rearrange, reduce
from fire import Fire
from peft import PeftModel
from torch.distributed.elastic.multiprocessing import errors
from torch.utils import data
from tqdm import tqdm
from transformers import LlavaConfig, LlavaProcessor

from data import (
    custom_collate_fn,
    get_cirr_image_dataset,
    get_cirr_text_dataset,
    get_coco_image_dataset,
    get_coco_text_dataset,
    get_fiq_image_dataset,
    get_fiq_text_dataset,
    get_flickr_image_dataset,
    get_flickr_text_dataset,
    recall_at_k,
)
from ft_llm import LlavaCustom

accelerator = Accelerator()


@dataclass
class Lazy:
    def __init__(self, func):
        self.func = func
        self.result = None

    def __call__(self):
        if self.result is None:
            self.result = self.func()
            self.func = None
        return self.result


@dataclass
class Modality:
    name: str
    dataset_name: str
    dataset: callable

    def __post_init__(self):
        self.dataset = Lazy(self.dataset)

    def on_embed_done(self, embs, indices):
        return embs, indices


class FIQQueryModality(Modality):
    def on_embed_done(self, embs, indices):
        embs = reduce(embs, "(b 2) d -> b d", "sum")
        indices = rearrange(indices, "(b e) -> e b", e=2)[0]
        return embs, indices


@dataclass
class Retrieval:
    ks: list[int]
    src_modality: Modality
    tgt_modality: Modality



def get_dataloader(dataset, transform):
    dataloader = data.DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=cpu_count() // accelerator.num_processes,
        collate_fn=lambda x: custom_collate_fn(x, transform),
        pin_memory=True,
        pin_memory_device=accelerator.device,
    )
    return accelerator.prepare(dataloader)


def map_to_embed(model, dataloader):
    model = model()
    embs = []
    indices = []
    if accelerator.is_main_process:
        dataloader = tqdm(dataloader)
    for batch in dataloader:
        data, index = batch
        with torch.inference_mode():
            emb = model(
                **data, output_hidden_states=True, return_dict=True
            ).hidden_states[-1][:, -1, :]
        emb = accelerator.gather_for_metrics(emb)
        embs.extend(emb)
        index = accelerator.gather_for_metrics(index)
        indices.extend(index)
    return torch.stack(embs), np.stack(indices)


def init_transform():
    transform = LlavaProcessor.from_pretrained("xtuner/llava-phi-3-mini-hf")
    transform.chat_template = "{% for message in messages %}{{ '<|' + message['role'] + '|>\n'}}{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}{{ '<image>' }}{% endfor %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{{ '\n' + content['text'] + '<|end|>\n' }}{% endfor %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}"
    transform.tokenizer.padding_side = "left"
    transform.tokenizer.padding = True
    model_cfg = LlavaConfig.from_pretrained("xtuner/llava-phi-3-mini-hf")
    transform.patch_size = model_cfg.vision_config.patch_size
    transform.vision_feature_select_strategy = model_cfg.vision_feature_select_strategy
    return transform


def init_model(lora_path):
    rank = accelerator.local_process_index
    model = LlavaCustom.from_pretrained(
        "xtuner/llava-phi-3-mini-hf",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=rank,
        attn_implementation="flash_attention_2",
    )
    if lora_path is not None:
        model = PeftModel.from_pretrained(
            model, lora_path, torch_device=f"cuda:{rank}"
        ).merge_and_unload()
    model = model.eval()
    model = accelerator.prepare(model)
    return model


def calculate_score(text_embs, img_embs):
    text_embs = F.normalize(text_embs, dim=-1)
    img_embs = F.normalize(img_embs, dim=-1)
    scores = text_embs @ img_embs.t()
    return scores


def calculate_pos_pairs(text_idx, img_idx):
    positive_pairs = torch.from_numpy(text_idx[:, None] == img_idx[None, :]).to(
        accelerator.device, non_blocking=True
    )
    return positive_pairs


def tee(metric_file_path, to_print):
    with open(metric_file_path, "a") as f:
        f.write(to_print)
    print(to_print, end="")


def modality_to_embed(transform, model, modality: Modality, lora_path):
    path_ = Path(lora_path) / "emb_cache"
    path_.mkdir(exist_ok=True)
    emb_path = path_ / f"{modality.dataset_name},{modality.name}.pt"
    idx_path = path_ / f"{modality.dataset_name},{modality.name}.npy"
    if emb_path.exists() and idx_path.exists():
        embs = torch.load(
            str(emb_path), map_location=accelerator.device, weights_only=True
        )
        idx = np.load(str(idx_path))
        if accelerator.is_main_process:
            print(f"Loaded `{modality.dataset_name}::{modality.name}` from cache.")
        return embs, idx

    if accelerator.is_main_process:
        print(f"Embedding `{modality.dataset_name}::{modality.name}`.")
    dataset = modality.dataset()
    dataloader = get_dataloader(dataset, transform)
    emb, idx = map_to_embed(model, dataloader)
    emb, idx = modality.on_embed_done(emb, idx)
    if accelerator.is_main_process:
        torch.save(emb, str(emb_path))
        np.save(str(idx_path), idx)
    return emb, idx


@errors.record
def main(lora_path: str = None):
    start = datetime.datetime.now()
    if not accelerator.is_main_process:
        transformers.utils.logging.disable_progress_bar()
        datasets.disable_progress_bars()

    transform = init_transform()

    flickr_text_modality = Modality(
        name="text",
        dataset_name="flickr",
        dataset=lambda: get_flickr_text_dataset(transform),
    )

    flickr_image_modality = Modality(
        name="image",
        dataset_name="flickr",
        dataset=lambda: get_flickr_image_dataset(transform),
    )

    coco_text_modality = Modality(
        name="text",
        dataset_name="coco",
        dataset=lambda: get_coco_text_dataset(transform),
    )

    coco_image_modality = Modality(
        name="image",
        dataset_name="coco",
        dataset=lambda: get_coco_image_dataset(transform),
    )

    fiq_dress_query_modality = FIQQueryModality(
        name="query",
        dataset_name="fiq_dress",
        dataset=lambda: get_fiq_text_dataset(transform, "dress"),
    )

    fiq_dress_image_modality = Modality(
        name="image",
        dataset_name="fiq_dress",
        dataset=lambda: get_fiq_image_dataset(transform, "dress"),
    )

    fiq_shirt_query_modality = FIQQueryModality(
        name="query",
        dataset_name="fiq_shirt",
        dataset=lambda: get_fiq_text_dataset(transform, "shirt"),
    )

    fiq_shirt_image_modality = Modality(
        name="image",
        dataset_name="fiq_shirt",
        dataset=lambda: get_fiq_image_dataset(transform, "shirt"),
    )

    fiq_toptee_query_modality = FIQQueryModality(
        name="query",
        dataset_name="fiq_toptee",
        dataset=lambda: get_fiq_text_dataset(transform, "toptee"),
    )

    fiq_toptee_image_modality = Modality(
        name="image",
        dataset_name="fiq_toptee",
        dataset=lambda: get_fiq_image_dataset(transform, "toptee"),
    )

    cirr_text_modality = Modality(
        name="query",
        dataset_name="cirr",
        dataset=lambda: get_cirr_text_dataset(transform),
    )

    cirr_image_modality = Modality(
        name="image",
        dataset_name="cirr",
        dataset=lambda: get_cirr_image_dataset(transform),
    )

    flickr_t2i_retrieval = Retrieval(
        ks=[1, 5, 10],
        src_modality=flickr_text_modality,
        tgt_modality=flickr_image_modality,
    )

    flickr_i2t_retrieval = Retrieval(
        ks=[1, 5, 10],
        src_modality=flickr_image_modality,
        tgt_modality=flickr_text_modality,
    )

    coco_t2i_retrieval = Retrieval(
        ks=[1, 5, 10],
        src_modality=coco_text_modality,
        tgt_modality=coco_image_modality,
    )

    coco_i2t_retrieval = Retrieval(
        ks=[1, 5, 10],
        src_modality=coco_image_modality,
        tgt_modality=coco_text_modality,
    )

    fiq_dress_retrieval = Retrieval(
        ks=[10, 50],
        src_modality=fiq_dress_query_modality,
        tgt_modality=fiq_dress_image_modality,
    )

    fiq_shirt_retrieval = Retrieval(
        ks=[10, 50],
        src_modality=fiq_shirt_query_modality,
        tgt_modality=fiq_shirt_image_modality,
    )

    fiq_toptee_retrieval = Retrieval(
        ks=[10, 50],
        src_modality=fiq_toptee_query_modality,
        tgt_modality=fiq_toptee_image_modality,
    )

    cirr_retrieval = Retrieval(
        ks=[1, 5, 10],
        src_modality=cirr_text_modality,
        tgt_modality=cirr_image_modality,
    )

    retrievals = [
        flickr_t2i_retrieval,
        flickr_i2t_retrieval,
        coco_t2i_retrieval,
        coco_i2t_retrieval,
        fiq_dress_retrieval,
        fiq_shirt_retrieval,
        fiq_toptee_retrieval,
        cirr_retrieval,
    ]

    model = Lazy(lambda x=lora_path: init_model(x))
    lora_path = lora_path or "metrics"
    os.makedirs(lora_path, exist_ok=True)

    for retrieval in retrievals:
        src_modality = retrieval.src_modality
        tgt_modality = retrieval.tgt_modality

        mod1_embs, mod1_idx = modality_to_embed(
            transform, model, src_modality, lora_path
        )

        mod2_embs, mod2_idx = modality_to_embed(
            transform, model, tgt_modality, lora_path
        )

        scores = calculate_score(mod1_embs, mod2_embs)
        positive_pairs = calculate_pos_pairs(mod1_idx, mod2_idx)

        metric_file_path = Path(lora_path) / "metrics.txt"
        if accelerator.is_main_process:
            src_modality_name = f"{src_modality.dataset_name}::{src_modality.name}"
            tgt_modality_name = f"{tgt_modality.dataset_name}::{tgt_modality.name}"
            to_print = f"{src_modality_name} -> {tgt_modality_name}\n"
            tee(metric_file_path, to_print)
        for k in retrieval.ks:
            recall = recall_at_k(scores, positive_pairs, k)
            recall = recall.mean().item()
            if accelerator.is_main_process:
                to_print = f"    R @ {k:2}: {recall:.4f}\n"
                tee(metric_file_path, to_print)

    if accelerator.is_main_process:
        end = datetime.datetime.now()
        duration = end - start
        print(f"Duration: {duration}")
    accelerator.end_training()


if __name__ == "__main__":
    Fire(main)
