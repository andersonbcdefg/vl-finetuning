import modal

python_version = "3.11"
flash_attn_version = "2.6.3"
pytorch_version = "2.7.1"
cuda_version = "12.8.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"
flash_attn_release = "flash_attn-2.6.3+cu128torch2.7-cp311-cp311-linux_x86_64.whl"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("git")
    .pip_install("torch==2.7.1")
    .run_commands(  # add flash-attn
        f"pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.9/{flash_attn_release}"
    )
    .pip_install(
        "transformers",
        "datasets",
        "torchvision",
        "qwen-vl-utils",
        "reportlab",
        "matplotlib",
        "numpy",
        "accelerate",
        "datasets",
        "trl",
        "peft",
        "timm",
        "pillow",
        "hf_xet",
        "torchao",
        "open_clip_torch",
        "pillow",
        "bitsandbytes"
    ).entrypoint([])
)


app = modal.App("seeclick-classifier")
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("seeclick-classifier-weights", create_if_missing=True)

@app.function(
    image=image,
    gpu="H100",
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/checkpoints": ckpt_vol
    },
    timeout=240 * 60
)
def train():
    #!/usr/bin/env python
    # fine_tune_regressor.py
    #
    # pip install "open_clip_torch>=2.24" datasets torch scikit-learn tqdm

    import os
    from pathlib import Path
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from datasets import load_dataset
    from tqdm.auto import tqdm
    import numpy as np
    from huggingface_hub import login

    # ----------------------------------------------------------------------
    # 0. Config ------------------------------------------------------------------
    DATASET = "andersonbcdefg/seeclick-10k-scored"   # repo id on the Hub
    SPLIT   = "train"                           # or whatever you named it
    BATCH   = 64
    EPOCHS  = 5
    LR      = 3e-4
    DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
    OUTDIR  = Path("/checkpoints")
    OUTDIR.mkdir(exist_ok=True)

    print("using device", DEVICE)

    # ----------------------------------------------------------------------
    # 1. Load model & pre‑processing ----------------------------------------
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16-SigLIP2-512", pretrained="webli"
    )
    model.eval().requires_grad_(False).to(DEVICE) # freeze backbone

    with torch.no_grad():
        h, w = model.visual.image_size          # e.g. (512, 512)
        dummy   = torch.zeros(1, 3, h, w, device=DEVICE)
        EMB_DIM = model.encode_image(dummy).shape[-1]   # 512 for SigLIP2‑512                # 512 for this model

    # ----------------------------------------------------------------------
    # 2. Build Dataset ------------------------------------------------------

    ds = load_dataset(DATASET, split=SPLIT).filter(lambda x: x['score'] is not None)          # returns 'image', 'score'
    ds = ds.shuffle(seed=42).train_test_split(test_size=0.1)
    train_ds, val_ds = ds["train"], ds["test"]

    def collate(batch):
        # → tensors: pixel_batch, score_batch
        imgs   = [preprocess(x["image"]) for x in batch]
        scores = torch.tensor([x["score"] for x in batch], dtype=torch.float32)
        return torch.stack(imgs), scores

    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                          num_workers=4, collate_fn=collate, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH*2, shuffle=False,
                          num_workers=4, collate_fn=collate, pin_memory=True)

    # ----------------------------------------------------------------------
    # 3. Regressor head -----------------------------------------------------

    class Regressor(nn.Module):
        def __init__(self, emb_dim: int):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(emb_dim, 256),
                nn.GELU(),
                nn.Linear(256, 1)           # output ∈ ℝ
            )
        def forward(self, x):               # x : (B, emb_dim)
            return self.mlp(x).squeeze(1)   # (B,)

    reg = Regressor(EMB_DIM).to(DEVICE)
    opt = torch.optim.AdamW(reg.parameters(), lr=LR)
    crit = nn.MSELoss()

    # ----------------------------------------------------------------------
    # 4. Train loop ---------------------------------------------------------

    def epoch_loop(loader, train: bool):
        reg.train(train)
        losses, preds, targs = [], [], []
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            pbar = tqdm(loader, leave=False)
            for px, y in pbar:
                px, y = px.to(DEVICE, non_blocking=True), y.to(DEVICE)
                with torch.no_grad():
                    emb = model.encode_image(px)       # (B, EMB_DIM)
                pred = reg(emb)                        # (B,)
                loss = crit(pred, y)
                if train:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                losses.append(loss.item())
                preds.append(pred.detach().cpu())
                targs.append(y.cpu())
                pbar.set_description(
                    f"{'train' if train else 'val'} loss {np.mean(losses):.4f}"
                )
        preds = torch.cat(preds)
        targs = torch.cat(targs)
        rmse  = torch.sqrt(nn.functional.mse_loss(preds, targs)).item()
        return np.mean(losses), rmse

    for ep in range(1, EPOCHS+1):
        train_loss, train_rmse = epoch_loop(train_dl, train=True)
        val_loss,   val_rmse   = epoch_loop(val_dl,   train=False)
        print(f"Epoch {ep:02d}: "
              f"train RMSE {train_rmse:.3f} | val RMSE {val_rmse:.3f}")
        torch.save(reg.state_dict(), OUTDIR / f"regressor_ep{ep:02d}.pt")

    print("Done. Latest regressor saved to", OUTDIR)
