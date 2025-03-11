import json
from pathlib import Path
from typing import List

import datasets

base_path = Path("~/datasets").expanduser()

_fiq_label_path = base_path / "fashion-iq" / "fashion-iq"


def _check_valid_arg(split, dress_types):
    valid_splits = ["test", "train", "val"]
    valid_dress_types = ["dress", "shirt", "toptee"]
    if split not in valid_splits:
        raise ValueError(f"split should be in {valid_splits}")
    if any(dress_type not in valid_dress_types for dress_type in dress_types):
        raise ValueError(f"dress_type should be in {valid_dress_types}")


def get_fiq_label_dataset(split: str, dress_types: List[str]):
    _check_valid_arg(split, dress_types)

    triplets: list[dict] = []
    for dress_type in dress_types:
        with open(_fiq_label_path / "captions" / f"cap.{dress_type}.{split}.json") as f:
            triplets.extend(json.load(f))

    dict_triplets = {k: [triplet[k] for triplet in triplets] for k in triplets[0]}
    return (
        datasets.Dataset.from_dict(dict_triplets)
        .rename_columns(
            {
                "candidate": "iq_id",
                "target": "it_id",
                "captions": "tq",
            }
        )
        .map(
            lambda x: {
                "it": list(map(map_id_to_path, x["it_id"])),
                "iq": list(map(map_id_to_path, x["iq_id"])),
            },
            batched=True,
        )
        .cast_column("it", datasets.Image())
        .cast_column("iq", datasets.Image())
    )


def get_fiq_image_dataset(split: str, dress_types: List[str]):
    _check_valid_arg(split, dress_types)

    list_: list[str] = []
    for dress_type in dress_types:
        with open(
            _fiq_label_path / "image_splits" / f"split.{dress_type}.{split}.json"
        ) as f:
            list_.extend(json.load(f))

    dict_triplets = {"id": list_}
    return (
        datasets.Dataset.from_dict(dict_triplets)
        .map(
            lambda x: {"image": list(map(map_id_to_path, x["id"]))},
            batched=True,
        )
        .cast_column("image", datasets.Image())
    )


def map_id_to_path(path):
    return str(base_path / "fashion-iq" / "images" / f"{path}.png")


if __name__ == "__main__":
    datasets.disable_caching()
    dset = (
        get_fiq_label_dataset("train", ["dress"])
        .map(
            lambda x: {
                "target": list(map(map_id_to_path, x["target"])),
                "candidate": list(map(map_id_to_path, x["candidate"])),
            },
            batched=True,
        )
        .cast_column("target", datasets.Image())
        .cast_column("candidate", datasets.Image())
    )
    print(dset)
    print(dset[0])
