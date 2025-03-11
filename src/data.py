from itertools import permutations

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset


def prompt_text(text):
    cont = [
        {"type": "text", "text": text},
    ]
    return prompt_user(cont)


def prompt_image_text(text):
    cont = [
        {"type": "image"},
        {"type": "text", "text": text},
    ]
    return prompt_user(cont)


def prompt_user(cont):
    msg = {"role": "user", "content": cont}
    return [msg]


def batch_apply_chat_template(transform, text):
    return {
        "text": transform.apply_chat_template(
            text,
            add_generation_prompt=True,
        ),
    }


def ir_text_map(x, ind, transform):
    text = [
        prompt_text(f"{y}\nSummary above sentence in one word:")
        for q in x["text"]
        for y in q
    ]
    return {
        **batch_apply_chat_template(transform, text),
        "index": [f"{i}" for i, q in zip(ind, x["text"]) for _ in q],
    }


def ir_image_map(x, ind, transform):
    text = [prompt_image_text("Summary above image in one word:") for _ in ind]
    return {
        **batch_apply_chat_template(transform, text),
        "index": [f"{y}" for y in ind],
    }


def fiq_dataset_map(x, ind, transform, style):
    tid = x["index"]
    img = x["images"]
    cap = x["text"]

    res_idx = []
    res_txt = []
    res_img = []
    for t, i, c in zip(tid, img, cap):
        for c_perm in permutations(c):
            res_idx.append(t)
            res_img.append(i)
            caption = ", ".join(cc.strip(".?, ") for cc in c_perm)
            caption = prompt_image_text(
                f"Change the style of this {style} to {caption}\nDescribe this modified {style} in one word based on its style:"
            )
            res_txt.append(caption)
    res_txt = transform.apply_chat_template(res_txt, add_generation_prompt=True)
    return {"index": res_idx, "images": res_img, "text": res_txt}


def cirr_text_map(x, ind, transform):
    text = [
        prompt_image_text(
            f'Modify this image with "{y}", describe modified image in one word:'
        )
        for y in x["text"]
    ]
    return batch_apply_chat_template(transform, text)


def fiq_image_map(x, ind, transform, style):
    text = [
        prompt_image_text(f"Describe this {style} in one word based on its style:")
        for _ in ind
    ]
    return batch_apply_chat_template(transform, text)


def cirr_image_map(x, ind, transform):
    text = [prompt_image_text("Describe this image in one word:") for _ in ind]
    return batch_apply_chat_template(transform, text)


def get_flickr_text_dataset(transform):
    return (
        load_dataset("royokong/flickr30k_test", split="test")
        .remove_columns("image")
        .map(
            lambda x, ind: ir_text_map(x, ind, transform),
            batched=True,
            with_indices=True,
        )
    )


def get_flickr_image_dataset(transform):
    return (
        load_dataset("royokong/flickr30k_test", split="test")
        .rename_column("image", "images")
        .map(
            lambda x, ind: ir_image_map(x, ind, transform),
            batched=True,
            with_indices=True,
        )
    )


def get_coco_text_dataset(transform):
    return (
        load_dataset("royokong/coco_test", split="test")
        .remove_columns("image")
        .map(
            lambda x, ind: ir_text_map(x, ind, transform),
            batched=True,
            with_indices=True,
        )
    )


def get_coco_image_dataset(transform):
    return (
        load_dataset("royokong/coco_test", split="test")
        .rename_column("image", "images")
        .map(
            lambda x, ind: ir_image_map(x, ind, transform),
            batched=True,
            with_indices=True,
        )
    )


def get_fiq_text_dataset(transform, style):
    return (
        load_dataset("royokong/fashioniq_val", split="val")
        .filter(lambda x: map(lambda y: y == style, x["category"]), batched=True)
        .remove_columns(["candidate_id", "category", "split", "target"])
        .rename_columns(
            {"candidate": "images", "caption": "text", "target_id": "index"}
        )
        .map(
            lambda x, ind: fiq_dataset_map(x, ind, transform, style),
            batched=True,
            with_indices=True,
        )
    )


def get_fiq_image_dataset(transform, style):
    return (
        load_dataset("royokong/fashioniq_val_imgs", split="val")
        .filter(lambda x: map(lambda y: y == style, x["category"]), batched=True)
        .remove_columns(["category", "split"])
        .rename_columns({"id": "index", "img": "images"})
        .map(
            lambda x, ind: fiq_image_map(x, ind, transform, style),
            batched=True,
            with_indices=True,
        )
    )


def get_cirr_text_dataset(transform):
    return (
        load_dataset("royokong/cirr_val", split="val")
        .remove_columns(["candidate_id", "group", "split", "target"])
        .rename_columns(
            {"target_id": "index", "candidate": "images", "caption": "text"}
        )
        .map(
            lambda x, ind: cirr_text_map(x, ind, transform),
            batched=True,
            with_indices=True,
        )
    )


def get_cirr_image_dataset(transform):
    return (
        load_dataset("royokong/cirr_imgs", split="val")
        .remove_columns(["category", "split"])
        .rename_columns({"id": "index", "img": "images"})
        .map(
            lambda x, ind: cirr_image_map(x, ind, transform),
            batched=True,
            with_indices=True,
        )
    )


def recall_at_k(scores, positive_pairs, k, transpose=False):
    dim = 0 if transpose else 1
    topk_indices = scores.topk(k, dim=dim).indices
    nb_true_positive = positive_pairs.sum(dim=dim)
    nb_retrieved_positive = positive_pairs.gather(dim, topk_indices).sum(dim=dim)
    recall = nb_retrieved_positive / nb_true_positive
    recall = (recall > 0).float()
    return recall * 100


def custom_collate_fn(batch, transform):
    coll = {}
    for key in batch[0]:
        coll[key] = [x[key] for x in batch]
    indices = coll.pop("index")
    return transform(
        **coll,
        return_tensors="pt",
        padding=True,
        padding_side="left",
    ), np.array(indices)


def get_cc3m_dataset():
    cc3m = load_dataset("pixparse/cc3m-wds", split="train")
    cc3m = cc3m.remove_columns(["__key__", "__url__"])
    return cc3m


def get_cc3m_dataset_with_img_embed():
    clip_tensor = torch.load("cc3m_clip_emb.pt", weights_only=True, map_location="cpu")
    cc3m = get_cc3m_dataset()
    train_data = WrapperDataset(cc3m, clip_tensor)
    return train_data


class WrapperDataset(Dataset):
    def __init__(self, ds, vec):
        super().__init__()
        self.ds = ds
        self.vec = vec
        assert len(ds) == len(vec)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        return {"clip_img_embed": self.vec[index], **self.ds[index]}
