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
    epochs: int = 100,
    batch: int = 16,
    imgsz: int = 640,
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

    root = Path("/tmp/yolo")
    train_img = root / "images" / "train"
    val_img = root / "images" / "val"
    train_lbl = root / "labels" / "train"
    val_lbl = root / "labels" / "val"
    for p in [train_img, val_img, train_lbl, val_lbl]:
        p.mkdir(parents=True, exist_ok=True)

    def _export(ds, img_dir: Path, lbl_dir: Path):
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
            x1, y1, x2, y2 = sample["bbox"]
            w, h = sample["size"]
            xc = ((x1 + x2) / 2) / w
            yc = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            with open(lbl_dir / f"{fname}.txt", "a") as f:
                f.write(f"0 {xc} {yc} {bw} {bh}\n")

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

    model = YOLO("yolo11n.pt")
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

