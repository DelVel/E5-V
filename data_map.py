import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPVisionModelWithProjection

from src.datasets.fashion_iq import get_fiq_image_dataset

accelerator = Accelerator()

processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

dataset = get_fiq_image_dataset("train", ["dress", "shirt", "toptee"])
dataloader = DataLoader(
    dataset,
    collate_fn=lambda x: processor(
        images=[xx["image"] for xx in x],
        return_tensors="pt",
        padding=True,
    ),
    batch_size=512,
    num_workers=16,
)
dataloader = accelerator.prepare(dataloader)


model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
model = accelerator.prepare(model)

embs = []
with torch.inference_mode():
    for a in tqdm(dataloader, disable=not accelerator.is_local_main_process):
        res = model(**a).image_embeds
        res = F.normalize(res, p=2, dim=-1)
        embs.extend(accelerator.gather_for_metrics(res))
embs = torch.stack(embs)
if accelerator.is_local_main_process:
    torch.save(embs, "fiq_clip_emb.pt")
    print(f"{len(embs)} saved.")

accelerator.end_training()
