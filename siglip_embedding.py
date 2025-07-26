from pathlib import Path
import io
import modal
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from PIL import Image
import datasets

from images import classifier_image as image

app = modal.App("siglip_embedding")
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("seeclick-classifier-weights", create_if_missing=True)

class Regressor(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 768),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(768, 768),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(768, 1),  # output ∈ ℝ
        )

    def forward(self, x):  # x : (B, emb_dim)
        return self.mlp(x).squeeze(1)  # (B,)

@app.cls(
    image=image,
    gpu="A10G",
    volumes={"/root/.cache/huggingface": hf_cache_vol, "/checkpoints": ckpt_vol},
    timeout=60,
    enable_memory_snapshot=True
)
class SigLIPEmbedding:
    @modal.enter(snap=True)
    def load_model(self):
        import torch
        import open_clip # type: ignore
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16-SigLIP2-512", pretrained="webli"
        )
        self.model = model
        self.preprocess = preprocess

    @modal.enter(snap=False)
    def setup(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @modal.batched(max_batch_size=64, wait_ms=425)
    def embed(self, inputs: list):

        # if bytes, encode as images
        if isinstance(inputs[0], bytes):
            pil_imgs = [Image.open(io.BytesIO(bts)) for bts in imgs]
            prepped = [self.preprocess(img) for img in pil_imgs] # type: ignore
            stacked = torch.stack(prepped) # type: ignore
            embs = self.model.encode_image( # type: ignore
                stacked.to(self.device)
            ).cpu().tolist()

        # if strings, encode as texts
        else:
            assert isinstance(inputs[0], str), "must pass bytes or str"
            embs = self.model.encode_text(inputs).cpu().tolist()

        return embs
