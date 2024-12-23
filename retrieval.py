import json
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from accelerate import Accelerator
from datasets import load_dataset, load_from_disk
from peft import PeftModel
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
)

from ft_llm import LlavaNextCustom
from datasets import disable_caching
from fire import Fire

accelerator = Accelerator()

llama3_template = "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n"


def recall_at_k(scores, positive_pairs, k):
    """
    Compute the recall at k for each sample
    :param scores: compability score between  text and image embeddings (nb texts, nb images)
    :param k: number of images to consider per text, for retrieval
    :param positive_pairs: boolean matrix of positive pairs (nb texts, nb images)
    :return: recall at k averaged over all texts
    """
    nb_texts, nb_images = scores.shape
    # for each text, sort according to image scores in decreasing order
    topk_indices = torch.topk(scores, k, dim=1)[1]
    # compute number of positives for each text
    nb_positive = positive_pairs.sum(dim=1)
    # nb_texts, k, nb_images
    topk_indices_onehot = torch.nn.functional.one_hot(
        topk_indices, num_classes=nb_images
    )
    # compute number of true positives
    positive_pairs_reshaped = positive_pairs.view(nb_texts, 1, nb_images)
    # a true positive means a positive among the topk
    nb_true_positive = (topk_indices_onehot * positive_pairs_reshaped).sum(dim=(1, 2))
    # compute recall at k
    return nb_true_positive / nb_positive


def batchify(batch_size, device, func, X, Y, *args, **kwargs):
    results = []
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        x = X[start:end].to(device)
        y = Y[start:end].to(device)
        result = func(x, y, *args, **kwargs).cpu()
        results.append(result)
    return torch.cat(results)


def emb_data(
    model,
    transform,
    dataset,
    device,
    emb_type="text",
    prompt=None,
    bsz=4,
    text_column="caption",
    img_column="img",
):
    # emb img
    def custom_collate_fn(batch):
        collated_batch = {}
        for key in batch[0].keys():
            collated_batch[key] = [b[key] for b in batch]
        return collated_batch

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=3 * bsz if emb_type == "text" else bsz,
        shuffle=False,
        num_workers=1,
        collate_fn=custom_collate_fn,
    )
    dataloader = accelerator.prepare(dataloader)
    embs = []
    bar = tqdm(total=len(dataloader))
    for batch in dataloader:
        if emb_type == "text":
            input_texts = [
                prompt.replace("<sent>", text)
                for text in sum(batch[text_column], start=[])
            ]
            inputs = transform(input_texts, return_tensors="pt", padding=True)
            for key in inputs:
                if inputs[key] is not None:
                    inputs[key] = inputs[key].to(device)
        else:
            input_texts = [prompt] * len(batch[img_column])
            inputs = transform(
                input_texts, batch[img_column], return_tensors="pt", padding=True
            ).to(device)

        with torch.no_grad():
            emb = model(
                **inputs, output_hidden_states=True, return_dict=True
            ).hidden_states[-1][:, -1, :]
            emb = F.normalize(emb, dim=-1)
        emb = accelerator.gather(emb)
        embs.append(emb.cpu().float())
        bar.update(1)
    embs = torch.cat(embs)
    total = 0
    for i in dataset:
        if emb_type == "text" and type(i[text_column]) is list:
            total += len(i[text_column])
        else:
            total += 1
    bar.close()
    return embs[:total]


def log_to_file(data, metrics, checkpoint_name, fiq_data_type=None):
    if data == "flickr30k" or data == "coco":
        output = f"{data}: {metrics['image_retrieval_recall@5']:.4f} {metrics['text_retrieval_recall@5']:.4f}"
    elif data == "fashioniq":
        assert len(metrics) == 2
        r_at_1, r_at_5 = metrics
        output = f"{data} {fiq_data_type}: R@10: {r_at_1:.4f} R@50: {r_at_5:.4f}"
    elif data == "cirr":
        assert len(metrics) == 3
        r_at_1, r_at_3, r_at_5 = metrics
        output = f"{data}:  R@1: {r_at_1:.4f} R@5: {r_at_3:.4f} R@10: {r_at_5:.4f}"
    else:
        raise ValueError(f"Unknown dataset {data}")

    if checkpoint_name is not None:
        with open(checkpoint_name, "a") as f:
            print(output, file=f)
    return output


def init_model_and_transform(lora_path):
    transform = LlavaNextProcessor.from_pretrained("llava-hf/llama3-llava-next-8b-hf")
    transform.tokenizer.padding_side = "left"
    transform.tokenizer.padding = True

    rank = dist.get_rank()
    with torch.cuda.device(rank):
        model = LlavaNextCustom.from_pretrained(
            "llava-hf/llama3-llava-next-8b-hf",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map=rank,
        )
        if lora_path is not None:
            model = PeftModel.from_pretrained(
                model, lora_path, torch_device=f"cuda:{rank}"
            ).merge_and_unload()

    return model, transform


def ir(model, transform, data, batch_size=None):
    img_prompt = llama3_template.format("<image>\nSummary above image in one word: ")
    text_prompt = llama3_template.format("<sent>\nSummary above sentence in one word: ")
    device = accelerator.device
    dataset = load_dataset(f"royokong/{data}_test", split="test")

    dataset = dataset.rename_column("text", "caption")
    dataset = dataset.rename_column("image", "img")
    if data == "coco":
        dataset = dataset.map(lambda x: {"caption": x["caption"][:5]}, num_proc=4)

    bsz = 4
    if batch_size is not None:
        bsz = batch_size

    text_embs = emb_data(
        model, transform, dataset, device, emb_type="text", prompt=text_prompt, bsz=bsz
    )
    img_embs = emb_data(
        model, transform, dataset, device, emb_type="image", prompt=img_prompt, bsz=bsz
    )

    texts_image_index = [i // 5 for i in range(img_embs.shape[0] * 5)]
    assert len(texts_image_index) == len(text_embs)

    assert text_embs.isnan().sum().item() == 0, "nan in retrieve emb"
    assert img_embs.isnan().sum().item() == 0, "nan in images emb"

    # get the score for each text and image pair
    scores = text_embs @ img_embs.t()

    positive_pairs = torch.zeros_like(scores, dtype=bool)
    positive_pairs[torch.arange(len(scores)), texts_image_index] = True
    metrics = {}
    recall_k_list = [1, 5, 10]
    batch_size = 64
    for recall_k in recall_k_list:
        # Recall@k in this implementation computes the **actual** recall (nb_true_positives/nb_positives).
        # For text retrieval, nb_positives represents all texts matching an image, often >1 due to multiple captions per image.
        # In contrast, CLIP-like papers define recall@k as binary: 1 if at least one match is found among the top-k, otherwise 0.
        # This can be derived from actual recall by checking if recall > 0.
        # Dataset-level recall is the average of per-instance recalls.
        metrics[f"image_retrieval_recall@{recall_k}"] = (
            (
                batchify(
                    batch_size, device, recall_at_k, scores, positive_pairs, k=recall_k
                )
                > 0
            )
            .float()
            .mean()
            .item()
        )
        metrics[f"text_retrieval_recall@{recall_k}"] = (
            (
                batchify(
                    batch_size,
                    device,
                    recall_at_k,
                    scores.T,
                    positive_pairs.T,
                    k=recall_k,
                )
                > 0
            )
            .float()
            .mean()
            .item()
        )

    return metrics


def cir(
    model,
    transform,
    img_prompt,
    text_img_prompt,
    data,
    fiq_data_type,
    device,
    fiq_two=False,
    batch_size=None,
):
    if data == "fashioniq":
        assert fiq_data_type in ["dress", "shirt", "toptee"]
        dataset = load_dataset("royokong/fashioniq_val")
        img_dataset = load_dataset("royokong/fashioniq_val_imgs")

        dataset = dataset["val"].filter(
            lambda x: x["category"] == fiq_data_type, num_proc=4
        )
        img_dataset = img_dataset["val"].filter(
            lambda x: x["category"] == fiq_data_type, num_proc=4
        )
    else:
        dataset = load_dataset("royokong/cirr_val")
        img_dataset = load_dataset("royokong/cirr_imgs")

        dataset = dataset["val"]
        img_dataset = img_dataset["val"]

    assert len(set(dataset["target_id"]) - set(img_dataset["id"])) == 0

    bsz = 4
    if fiq_two:
        bsz //= 2
    if batch_size is not None:
        bsz = batch_size

    # emb img
    def custom_collate_fn(batch):
        collated_batch = {}
        for key in batch[0].keys():
            collated_batch[key] = [b[key] for b in batch]
        return collated_batch

    collate_fn = custom_collate_fn

    img_dataloader = torch.utils.data.DataLoader(
        img_dataset, batch_size=bsz, shuffle=False, num_workers=4, collate_fn=collate_fn
    )
    img_dataloader = accelerator.prepare(img_dataloader)
    images_embs = []
    bar = tqdm(total=len(img_dataloader))
    for batch in img_dataloader:
        input_texts = [img_prompt] * len(batch["img"])
        inputs = transform(
            input_texts, batch["img"], return_tensors="pt", padding=True
        ).to(device)
        with torch.no_grad():
            embs = model(
                **inputs, output_hidden_states=True, return_dict=True
            ).hidden_states[-1][:, -1, :]
            embs = F.normalize(embs, dim=-1)
            assert embs.isnan().sum() == 0, "nan in emb after norm"
        embs = accelerator.gather(embs)
        images_embs.append(embs.cpu().float())
        bar.update(1)
    images_emb = torch.cat(images_embs)[: len(img_dataset["id"])]
    images_ids = img_dataset["id"]

    bar.close()

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=bsz, shuffle=False, num_workers=4, collate_fn=collate_fn
    )

    retrieve_emb = []
    dataloader = accelerator.prepare(dataloader)
    bar = tqdm(total=len(dataloader))
    for batch in dataloader:
        images = batch["candidate"]
        if data == "fashioniq":
            caption = batch["caption"]
            if fiq_two:
                caption = caption + [i[::-1] for i in caption]
                images = images + images
            input_texts = [
                text_img_prompt.replace(
                    "<sent>", ", ".join([cc.strip(".?, ") for cc in c])
                )
                for c in caption
            ]
        else:
            input_texts = [
                text_img_prompt.replace("<sent>", c) for c in batch["caption"]
            ]

        inputs = transform(input_texts, images, return_tensors="pt", padding=True).to(
            device
        )
        with torch.no_grad():
            embs = model(
                **inputs, output_hidden_states=True, return_dict=True
            ).hidden_states[-1][:, -1, :]
            if fiq_two:
                embs = embs[: len(batch["caption"])] + embs[len(batch["caption"]) :]
            embs = F.normalize(embs, dim=-1)
        embs = accelerator.gather(embs)
        retrieve_emb.append(embs.cpu().float())
        bar.update(1)
    retrieve_emb = torch.cat(retrieve_emb)[: len(dataset["target_id"])]
    target_ids = dataset["target_id"]
    bar.close()

    assert retrieve_emb.isnan().sum().item() == 0, "nan in retrieve emb"
    assert images_emb.isnan().sum().item() == 0, "nan in images emb"

    scores = retrieve_emb @ images_emb.t()

    labels = []
    for i, target_id in enumerate(target_ids):
        labels.append(images_ids.index(target_id))

    if data == "cirr":
        # remove reference itself like SEARLE
        mask_index = [images_ids.index(label) for label in dataset["candidate_id"]]
        for i, mid in enumerate(mask_index):
            scores[i][mid] = -1

    def cir_recall_at_k(scores, labels, k):
        """
        Calculate Recall@k using PyTorch
        """
        num_queries = scores.size(0)
        recalls = []
        for i in range(num_queries):
            top_k_indices = torch.topk(scores[i], k=k, largest=True).indices
            recalls.append(int(labels[i] in top_k_indices))
        return sum(recalls) / num_queries

    if data == "fashioniq":
        # Calculate R@1, R@3, and R@5
        r_at_1 = cir_recall_at_k(scores, labels, 10)
        r_at_5 = cir_recall_at_k(scores, labels, 50)
        metrics = [r_at_1, r_at_5]
    else:
        # Calculate R@1, R@3, and R@5
        r_at_1 = cir_recall_at_k(scores, labels, 1)
        r_at_3 = cir_recall_at_k(scores, labels, 5)
        r_at_5 = cir_recall_at_k(scores, labels, 10)
        metrics = [r_at_1, r_at_3, r_at_5]

    return metrics


def main(
    lora_path: str = None,
    batch_size: int = 2,
):
    if os.environ.get("NCCL_DEBUG", None) is None:
        os.environ["NCCL_DEBUG"] = "ERROR"

    device = accelerator.device

    model, transform = init_model_and_transform(lora_path)
    model.to(device)

    disable_caching()

    datasets = [
        "flickr30k",
        "coco",
        "fashioniq dress",
        "fashioniq shirt",
        "fashioniq toptee",
        "cirr",
    ]

    all_results = []
    for data in datasets:
        if "fashioniq" in data:
            data, fiq_data_type = data.split(" ")
            fiq_two = True
        else:
            fiq_data_type = None
            fiq_two = False

        if data == "flickr30k" or data == "coco":
            metrics = ir(model, transform, data, batch_size)
        elif data == "fashioniq" or data == "cirr":
            metrics = cir_2(batch_size, data, fiq_data_type, fiq_two, model, transform)
        else:
            raise ValueError(f"Unknown dataset {data}")

        if accelerator.is_main_process:
            print(metrics)
            if lora_path is not None:
                checkpoint_name = lora_path.replace("/", "_") + ".txt"
            else:
                checkpoint_name = None
            all_results.append(
                log_to_file(data, metrics, checkpoint_name, fiq_data_type=fiq_data_type)
            )

    if accelerator.is_main_process:
        print("\n".join(all_results))


def cir_2(batch_size, data, fiq_data_type, fiq_two, model, transform):
    if data == "fashioniq":
        fiq_data_name = fiq_data_type
        if fiq_data_type == "toptee":
            fiq_data_name = "shirt"
        img_prompt = (
            f"<image>\n Describe this {fiq_data_name} in one word based on its style:"
        )
        text_img_prompt = f"<image> change the style of this {fiq_data_name} to <sent>\n Describe this modified {fiq_data_name} in one word based on its style:"
    else:
        img_prompt = "<image>\n Describe this image in one word:"
        text_img_prompt = '<image>Modify this image with "<sent>", describe modified image in one word:'
    img_prompt = llama3_template.format(img_prompt)
    text_img_prompt = llama3_template.format(text_img_prompt)

    return cir(
        model,
        transform,
        img_prompt,
        text_img_prompt,
        data,
        fiq_data_type,
        accelerator.device,
        fiq_two=fiq_two,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    Fire(main)
