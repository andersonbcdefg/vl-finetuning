from pathlib import Path

import modal
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from images import classifier_image as image

app = modal.App("seeclick-classifier")
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


@app.function(
    image=image,
    gpu="H100",
    volumes={"/root/.cache/huggingface": hf_cache_vol, "/checkpoints": ckpt_vol},
    timeout=240 * 60,
)
def train():
    import open_clip

    # ----------------------------------------------------------------------
    # 0. Config ------------------------------------------------------------------
    DATASET = "andersonbcdefg/seeclick-10k-scored"  # repo id on the Hub
    SPLIT = "train"  # or whatever you named it
    BATCH = 256
    EPOCHS = 100
    LR = 3e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    OUTDIR = Path("/checkpoints")
    OUTDIR.mkdir(exist_ok=True)
    CACHE_TRAIN = OUTDIR / "train_embeds.pt"
    CACHE_VAL = OUTDIR / "val_embeds.pt"
    EMB_DIM = 768

    print("using device", DEVICE)

    # ----------------------------------------------------------------------
    # 1. Load model & pre‑processing ----------------------------------------

    # ----------------------------------------------------------------------
    def collate(batch):
        imgs = [preprocess(x["image"]) for x in batch]
        scores = torch.tensor([x["score"] for x in batch], dtype=torch.float32)
        return torch.stack(imgs), scores

    def embed_split(model, raw_split, desc):
        dl = DataLoader(
            raw_split,
            batch_size=BATCH,
            shuffle=False,
            num_workers=4,
            collate_fn=collate,
            pin_memory=True,
        )
        embs, ys = [], []
        with torch.no_grad():
            for px, y in tqdm(dl, desc=f"embedding {desc}", leave=False):
                emb = model.encode_image(px.to(DEVICE, non_blocking=True)).cpu()
                embs.append(emb)
                ys.append(y)
        return torch.cat(embs), torch.cat(ys)

    if CACHE_TRAIN.exists() and CACHE_VAL.exists():
        # ----------- load ---------------------------------------------------
        train_emb, train_scores = torch.load(CACHE_TRAIN)
        val_emb, val_scores = torch.load(CACHE_VAL)
        print("loaded cached embeddings from /checkpoints")
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16-SigLIP2-512", pretrained="webli"
        )
        model = torch.compile(model, mode="reduce-overhead")
        assert isinstance(model, nn.Module)
        model.eval().requires_grad_(False).to(DEVICE)  # freeze backbone

        # ----------- compute & save ----------------------------------------
        ds = load_dataset(DATASET, split=SPLIT)
        scores = list(ds.select_columns(["score"]))
        valid = [i for i, x in enumerate(scores) if x["score"] is not None]
        ds = ds.select(valid).shuffle(seed=42).train_test_split(test_size=0.1)
        raw_train, raw_val = ds["train"], ds["test"]

        train_emb, train_scores = embed_split(model, raw_train, "train")
        val_emb, val_scores = embed_split(model, raw_val, "val")

        torch.save((train_emb, train_scores), CACHE_TRAIN)
        torch.save((val_emb, val_scores), CACHE_VAL)
        print("saved embeddings to /checkpoints")

    train_ds = torch.utils.data.TensorDataset(train_emb, train_scores)
    val_ds = torch.utils.data.TensorDataset(val_emb, val_scores)

    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH * 2, shuffle=False, pin_memory=True)

    # ----------------------------------------------------------------------
    # 3. Head --------------------------------------------------------------------
    reg = Regressor(EMB_DIM).to(DEVICE)
    opt = torch.optim.AdamW(reg.parameters(), lr=LR)
    crit = nn.MSELoss()

    def epoch_loop(loader, train: bool):
        reg.train(train)
        losses, preds, accs, targs = [], [], [], []
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for emb, y in loader:
                emb, y = emb.to(DEVICE, non_blocking=True), y.to(DEVICE)
                pred = reg(emb)
                loss = crit(pred, y)
                acc = (
                    (torch.round(torch.clamp(pred, min=0.0, max=5.0)) == y)
                    .float()
                    .mean()
                )
                if train:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                losses.append(loss.item())
                accs.append(acc.item())
                preds.append(pred.detach().cpu())
                targs.append(y.cpu())

        preds, targs = torch.cat(preds), torch.cat(targs)
        rmse = torch.sqrt(nn.functional.mse_loss(preds, targs)).item()
        return np.mean(losses), rmse, np.mean(accs)

    # ----------------------------------------------------------------------
    # 4. Training ---------------------------------------------------------------
    for ep in range(1, EPOCHS + 1):
        train_loss, train_rmse, train_acc = epoch_loop(train_dl, train=True)
        val_loss, val_rmse, val_acc = epoch_loop(val_dl, train=False)
        print(f"Epoch {ep:02d}: train RMSE {train_rmse:.3f} | val RMSE {val_rmse:.3f}")
        print(f"Train acc: {train_acc:.4f} | Val acc: {val_acc:.4f}")
        torch.save(reg.state_dict(), OUTDIR / f"regressor_ep{ep:02d}.pt")

    print("Done. Latest regressor saved to", OUTDIR)


@app.function(
    image=image,
    gpu="H100",
    volumes={"/root/.cache/huggingface": hf_cache_vol, "/checkpoints": ckpt_vol},
    timeout=240 * 60,
)
def inference():
    import open_clip

    # ----------------------------------------------------------------------
    # 0. Config ------------------------------------------------------------------
    DATASET = "andersonbcdefg/seeclick-10k-scored"  # repo id on the Hub
    SPLIT = "train"  # or whatever you named it
    BATCH = 256
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    OUTDIR = Path("/checkpoints")
    OUTDIR.mkdir(exist_ok=True)
    EMB_DIM = 768

    print("using device", DEVICE)

    # ----------------------------------------------------------------------
    # 1. Load model & pre‑processing ----------------------------------------

    # ----------------------------------------------------------------------
    def collate(batch):
        imgs = [preprocess(x["image"]) for x in batch]
        scores = torch.tensor([x["score"] for x in batch], dtype=torch.float32)
        return torch.stack(imgs), scores

    def embed_split(model, raw_split, desc):
        dl = DataLoader(
            raw_split,
            batch_size=BATCH,
            shuffle=False,
            num_workers=4,
            collate_fn=collate,
            pin_memory=True,
        )
        embs, ys = [], []
        with torch.no_grad():
            for px, y in tqdm(dl, desc=f"embedding {desc}", leave=False):
                emb = model.encode_image(px.to(DEVICE, non_blocking=True)).cpu()
                embs.append(emb)
                ys.append(y)
        return torch.cat(embs), torch.cat(ys)

    if CACHE_TRAIN.exists() and CACHE_VAL.exists():
        # ----------- load ---------------------------------------------------
        train_emb, train_scores = torch.load(CACHE_TRAIN)
        val_emb, val_scores = torch.load(CACHE_VAL)
        print("loaded cached embeddings from /checkpoints")
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16-SigLIP2-512", pretrained="webli"
        )
        model = torch.compile(model, mode="reduce-overhead")
        assert isinstance(model, nn.Module)
        model.eval().requires_grad_(False).to(DEVICE)  # freeze backbone

        # ----------- compute & save ----------------------------------------
        ds = load_dataset(DATASET, split=SPLIT)
        scores = list(ds.select_columns(["score"]))
        valid = [i for i, x in enumerate(scores) if x["score"] is not None]
        ds = ds.select(valid).shuffle(seed=42).train_test_split(test_size=0.1)
        raw_train, raw_val = ds["train"], ds["test"]

        train_emb, train_scores = embed_split(model, raw_train, "train")
        val_emb, val_scores = embed_split(model, raw_val, "val")

        torch.save((train_emb, train_scores), CACHE_TRAIN)
        torch.save((val_emb, val_scores), CACHE_VAL)
        print("saved embeddings to /checkpoints")

    train_ds = torch.utils.data.TensorDataset(train_emb, train_scores)
    val_ds = torch.utils.data.TensorDataset(val_emb, val_scores)

    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH * 2, shuffle=False, pin_memory=True)

    # ----------------------------------------------------------------------
    # 3. Head --------------------------------------------------------------------
    reg = Regressor(EMB_DIM).to(DEVICE)
    opt = torch.optim.AdamW(reg.parameters(), lr=LR)
    crit = nn.MSELoss()

    def epoch_loop(loader, train: bool):
        reg.train(train)
        losses, preds, accs, targs = [], [], [], []
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for emb, y in loader:
                emb, y = emb.to(DEVICE, non_blocking=True), y.to(DEVICE)
                pred = reg(emb)
                loss = crit(pred, y)
                acc = (
                    (torch.round(torch.clamp(pred, min=0.0, max=5.0)) == y)
                    .float()
                    .mean()
                )
                if train:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                losses.append(loss.item())
                accs.append(acc.item())
                preds.append(pred.detach().cpu())
                targs.append(y.cpu())

        preds, targs = torch.cat(preds), torch.cat(targs)
        rmse = torch.sqrt(nn.functional.mse_loss(preds, targs)).item()
        return np.mean(losses), rmse, np.mean(accs)

    # ----------------------------------------------------------------------
    # 4. Training ---------------------------------------------------------------
    for ep in range(1, EPOCHS + 1):
        train_loss, train_rmse, train_acc = epoch_loop(train_dl, train=True)
        val_loss, val_rmse, val_acc = epoch_loop(val_dl, train=False)
        print(f"Epoch {ep:02d}: train RMSE {train_rmse:.3f} | val RMSE {val_rmse:.3f}")
        print(f"Train acc: {train_acc:.4f} | Val acc: {val_acc:.4f}")
        torch.save(reg.state_dict(), OUTDIR / f"regressor_ep{ep:02d}.pt")

    print("Done. Latest regressor saved to", OUTDIR)
