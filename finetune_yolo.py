from __future__ import annotations

from pathlib import Path
import base64
import modal

from images import yolo_image as _yolo_image

# include local utils for data loading
image = _yolo_image.add_local_python_source("vl_utils")

app = modal.App("finetune-yolo-ui")

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("yolo-checkpoints", create_if_missing=True)

def iou(a, b):                     # a, b = (x1, y1, x2, y2) in *relative* coords
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b
    inter_w  = max(0, min(xa2, xb2) - max(xa1, xb1))
    inter_h  = max(0, min(ya2, yb2) - max(ya1, yb1))
    inter    = inter_w * inter_h
    union    = (xa2-xa1)*(ya2-ya1) + (xb2-xb1)*(yb2-yb1) - inter
    return inter / union if union else 0.0

@app.function(
    image=image,
    gpu="H100",
    volumes={"/root/.cache/huggingface": hf_cache_vol, "/checkpoints": ckpt_vol},
    secrets=[modal.Secret.from_name("HF-SECRET")],
    timeout=60 * 240,
)
def train(
    run_name: str,
    train_datasets: str,
    epochs: int = 10,
    batch: int = 16,
    imgsz: int = 768,
    test_size: int = 500,
):
    """Fine‑tune a YOLO detector on the specified datasets.

    Args:
        run_name: Name for checkpoint directory.
        train_datasets: Comma separated names from ``vl_utils.data.DATASETS``.
        epochs: Number of training epochs.
        batch: Batch size for YOLO trainer.
        imgsz: Image size for YOLO.
        test_size: Number of samples held out for validation.
    """
    from ultralytics import YOLO
    from vl_utils.data import load_data

    train_ds, val_ds = load_data(train_datasets.split(","), test_size, seed=42)

    def _unique_imgs(ds):
        return len({s["data_uri"] for s in ds})

    n_train = _unique_imgs(train_ds)
    n_val   = _unique_imgs(val_ds) if val_ds else 0
    print(f"Dataset prepared!\n→ {n_train} train / {n_val} val images\n→ {len(train_ds)} train boxes, {len(val_ds) if val_ds else 0} val boxes")

    root = Path("/tmp/yolo")
    train_img = root / "images" / "train"
    val_img = root / "images" / "val"
    train_lbl = root / "labels" / "train"
    val_lbl = root / "labels" / "val"
    for p in [train_img, val_img, train_lbl, val_lbl]:
        p.mkdir(parents=True, exist_ok=True)

    def _export(ds, img_dir: Path, lbl_dir: Path):
        last_uri = None
        kept = []
        mapping: dict[str, str] = {}
        for sample in ds:
            uri: str = sample["data_uri"]
            fname = mapping.get(uri)
            if fname is None:
                fname = f"{len(mapping):06d}"
                mapping[uri] = fname
                header, b64 = uri.split(",", 1)
                img_bytes = base64.b64decode(b64)
                with open(img_dir / f"{fname}.jpg", "wb") as f:
                    f.write(img_bytes)

            if uri != last_uri:
                last_uri = uri
                kept = []

            if not kept or all(iou(sample['bbox'], k) < 0.9 for k in kept):
                kept.append(sample['bbox'])
                x1, y1, x2, y2 = sample["bbox"]
                w, h = sample["size"]

                x1, x2 = x1 / w, x2 / w
                y1, y2 = y1 / h, y2 / h
                xc = (x1 + x2) / 2
                yc = (y1 + y2) / 2
                bw =  x2 - x1
                bh =  y2 - y1

                assert 0 <= min(xc, yc, bw, bh) <= max(xc, yc, bw, bh) <= 1, f"bbox out of range: {xc, yc, bw, bh}. orig bbox: {sample['bbox']}"

                with open(lbl_dir / f"{fname}.txt", "a") as f:
                    f.write(f"0 {xc} {yc} {bw} {bh}\n")

            else:
                print("removing near-duplicate box")

    _export(train_ds, train_img, train_lbl)
    if val_ds is not None:
        _export(val_ds, val_img, val_lbl)

    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        f"""path: {root}
train: images/train
val: images/val
names:
  0: ui
"""
    )

    model = YOLO("yolo11m.pt")
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project="/checkpoints",
        name=run_name,
        exist_ok=True,
    )


@app.function(
    image=image,
    gpu="H100",
    volumes={"/checkpoints": ckpt_vol},
    timeout=60 * 15,
)
def infer(run_name: str, image_path: str):
    """Run inference using a trained checkpoint on ``image_path``.

    Saves an annotated image alongside raw predictions."""
    from ultralytics import YOLO
    from PIL import Image

    ckpt = Path("/checkpoints") / run_name / "weights" / "best.pt"
    model = YOLO(str(ckpt))
    img = Image.open(image_path).convert("RGB")
    results = model.predict(img)
    print(results[0].boxes.xyxy)
    out_path = Path("/checkpoints") / run_name / "inference.jpg"
    results[0].save(filename=str(out_path))
    print("Saved annotated image to", out_path)
