from pathlib import Path

import modal

from image import image

app = modal.App("screenspot-improved")

volume = modal.Volume.from_name("marked-images", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/root/.cache/huggingface": hf_cache_vol, "/outputs": volume},
)
def main():
    import csv
    from typing import List, Tuple

    from datasets import load_dataset
    from PIL import Image, ImageDraw, ImageFont  # PIL handles alpha better than cv2
    from ultralytics import YOLO

    MODEL_PATH = "/yolo_model.pt"  # copied into the image at build time
    OUTPUT_DIR = Path("/outputs")  # Mount a volume or modal.Dataset if you wish
    CONF_THRESH = 0.6  # whatever works for your model
    FILL_ALPHA = 80  # 0‑255   (~31 % opacity; adjust to taste)
    STROKE_WIDTH = 2

    PALETTE = [
        "#E76F51",
        "#F4A261",
        "#2A9D8F",
        "#E9C46A",
        "#264653",
        "#8AB17D",
        "#457B9D",
        "#A8DADC",
        "#DDA15E",
        "#6D597A",
    ]

    model = YOLO(MODEL_PATH)
    dataset = load_dataset("andersonbcdefg/seeclick-10k-scored", split="train")

    def _color_for_class(cls_idx: int) -> str:
        """Pick a color from the palette, repeat if > len(PALETTE) classes."""
        return PALETTE[cls_idx % len(PALETTE)]

    def _rgba(hex_rgb: str, alpha: int = 80) -> tuple[int, int, int, int]:
        """Convert '#RRGGBB' to (r,g,b,a) with given transparency."""
        hex_rgb = hex_rgb.lstrip("#")
        r, g, b = tuple(int(hex_rgb[i : i + 2], 16) for i in (0, 2, 4))
        return (r, g, b, alpha)

    def _text_size(font, text):
        try:  # Pillow ≥10
            l, t, r, b = font.getbbox(text)
            return r - l, b - t
        except AttributeError:  # Pillow <10
            return font.getsize(text)

    def _label_xy(x1, y1, x2, y2, tw, th, W, H, margin=2):
        """
        Choose (tx, ty) so that the label rectangle
        (tx, ty, tx+tw, ty+th) is fully inside the image.

        Preference order:
          1. just above the box
          2. just below the box
          3. inside the box, top‑left
        """
        # 1️⃣ above
        tx, ty = x1 + margin, y1 - th - margin
        if ty >= 0:
            return tx, ty

        # 2️⃣ below
        ty = y2 + margin
        if ty + th <= H:
            return tx, ty

        # 3️⃣ fallback inside
        ty = y1 + margin
        # horizontal clamp so we never run past the right edge
        tx = min(tx, W - tw - margin)
        return tx, ty

    def _annotate_one(img_path: Path, result) -> List[Tuple[str, str, List[int], str]]:
        """
        Draw boxes for a single Result object, save overlay PNG,
        and return rows for the CSV.  Uses existing helper funcs.
        """
        im = Image.open(img_path).convert("RGB")

        overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
        draw_overlay = ImageDraw.Draw(overlay)
        draw_text = ImageDraw.Draw(overlay)
        font = ImageFont.load_default()

        W, H = im.size
        rows = []
        # ➊ collect and sort detections in raster‑scan order
        sorted_boxes = sorted(
            result.boxes,
            key=lambda b: (int(b.xyxy[0][1]), int(b.xyxy[0][0])),  # (y1, x1)
        )

        # ➋ label / draw / record exactly in that order
        for i, box in enumerate(sorted_boxes, start=1):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label_text = str(i)
            color = _color_for_class(i)

            # semi‑transparent box
            draw_overlay.rectangle(
                [(x1, y1), (x2, y2)],
                fill=_rgba(color, FILL_ALPHA),  # ← more visible
                outline=color,
                width=STROKE_WIDTH,
            )

            tw, th = _text_size(font, label_text)
            tx, ty = _label_xy(x1, y1, x2, y2, tw, th, W, H)

            # label background + text
            draw_overlay.rectangle(
                [(tx - 2, ty - 2), (tx + tw + 2, ty + th + 2)],
                fill=color,
                outline=color,
            )
            draw_text.text((tx, ty), label_text, fill="#FFFFFF", font=font)

            rows.append((img_path.name, label_text, [x1, y1, x2, y2], color))

        out_path = OUTPUT_DIR / f"{img_path.stem}_boxes.png"
        im_out = Image.alpha_composite(im.convert("RGBA"), overlay).convert("RGB")
        im_out.save(out_path, format="PNG")

        return rows

    def _annotate_one_pil(
        img: Image.Image,
        file_name: str,
        result,
        font: ImageFont.FreeTypeFont,
    ) -> List[Tuple[str, str, List[int], str]]:
        """
        Draws raster‑scan–ordered boxes + numeric labels on `img`,
        writes `<stem>_boxes.png` into OUTPUT_DIR, and returns rows
        for metadata.csv.

        Args
        ----
        img : PIL.Image.Image
            The original screenshot (RGB).
        file_name : str
            Logical name for CSV / output image (e.g. "shot_001.png").
        result : ultralytics.engine.results.Results
            The detection result returned by YOLO for this image.
        font : PIL.ImageFont.FreeTypeFont
            Font object to use for label text.

        Returns
        -------
        List[Tuple[str, str, List[int], str]]
            (file_name, label_text, [x1,y1,x2,y2], color_hex)
            in the same order the labels appear on the image.
        """
        W, H = img.size
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw_overlay = ImageDraw.Draw(overlay)
        draw_text = ImageDraw.Draw(overlay)

        rows: List[Tuple[str, str, List[int], str]] = []

        # --- raster‑scan sort ---------------------------------------------------
        boxes_sorted = sorted(
            result.boxes,
            key=lambda b: (int(b.xyxy[0][1]), int(b.xyxy[0][0])),  # (y1, x1)
        )

        # --- draw each detection -----------------------------------------------
        for i, box in enumerate(boxes_sorted, start=1):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label_text = str(i)
            color_hex = _color_for_class(i)

            # translucent fill + outline
            draw_overlay.rectangle(
                [(x1, y1), (x2, y2)],
                fill=_rgba(color_hex, FILL_ALPHA),
                outline=color_hex,
                width=STROKE_WIDTH,
            )

            # label placement
            tw, th = _text_size(font, label_text)
            tx, ty = _label_xy(x1, y1, x2, y2, tw, th, W, H)

            draw_overlay.rectangle(
                [(tx - 2, ty - 2), (tx + tw + 2, ty + th + 2)],
                fill=color_hex,
                outline=color_hex,
            )
            draw_text.text((tx, ty), label_text, fill="#FFFFFF", font=font)

            rows.append((file_name, label_text, [x1, y1, x2, y2], color_hex))

        # --- save annotated image ----------------------------------------------
        out_img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
        out_path = OUTPUT_DIR / f"{Path(file_name).stem}_boxes.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_img.save(out_path, format="PNG")

        return rows

    def process_folder(folder: Path, batch_size: int = 16):
        """
        Run detection in batches on GPU and write annotated images + metadata.csv.
        """
        imgs = sorted(
            p for p in folder.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        rows = []

        for start in range(0, len(imgs), batch_size):
            batch_paths = imgs[start : start + batch_size]

            # `model.predict` will push the batch to GPU once and split it internally
            results = model.predict(
                batch_paths,
                conf=CONF_THRESH,
                batch=batch_size,  # actual batch size on GPU
                verbose=False,
            )

            # zip back each result with its path
            for img_path, r in zip(batch_paths, results):
                # r is a single Ultralytics Result object
                rows.extend(
                    _annotate_one(img_path, r)
                )  # ← writes PNG, returns CSV row(s)

        # dump CSV once at end
        with (OUTPUT_DIR / "metadata.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["file_name", "label", "box_xyxy", "color_hex"])
            writer.writerows(rows)

    def _run_and_save_batch(
        imgs: List[Image.Image],
        names: List[str],  # display / csv names, same order as imgs
        model,
        font,
        overlay_helpers,
    ) -> List[Tuple[str, str, List[int], str]]:
        """
        Pushes `imgs` through YOLO once, draws overlays, writes PNGs,
        returns metadata rows.  `overlay_helpers` is a namespace with
        _rgba, _color_for_class, _text_size, _label_xy, etc.
        """
        results = model.predict(imgs, conf=CONF_THRESH, verbose=False, batch=len(imgs))
        rows = []

        for img, name, res in zip(imgs, names, results):
            # leverage existing _annotate_one but pass PIL instead of a path
            rows.extend(_annotate_one_pil(img, name, res, font))
        return rows

    def process_hf_dataset(
        dataset_name: str,
        split: str = "train",
        batch_size: int = 16,
    ):
        """
        Streams an image dataset from Hugging Face, runs GPU batches, writes:
          /outputs/<name>_boxes.png
          /outputs/metadata.csv
        """
        ds = load_dataset(dataset_name, split=split, streaming=True)
        num_batches = 10_000 // batch_size + 1
        batches_so_far = 0
        batch_imgs, batch_names, all_rows = [], [], []
        font = ImageFont.load_default()

        for i, sample in enumerate(ds):
            img = sample["image"]  # PIL.Image
            fname = f"{i}.png"
            batch_imgs.append(img)
            batch_names.append(fname)

            if len(batch_imgs) == batch_size:
                all_rows += _run_and_save_batch(
                    batch_imgs, batch_names, model, font, globals()
                )
                batch_imgs, batch_names = [], []
                batches_so_far += 1
                print(f"Processed {batches_so_far}/{num_batches} batches")

        # leftover
        if batch_imgs:
            all_rows += _run_and_save_batch(
                batch_imgs, batch_names, model, font, globals()
            )
            batches_so_far += 1
            print(f"Processed {batches_so_far}/{num_batches} batches")

        # write CSV once
        csv_path = OUTPUT_DIR / "metadata.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["file_name", "label", "box_xyxy", "color_hex"])
            writer.writerows(all_rows)

    process_hf_dataset("andersonbcdefg/seeclick-10k-scored")
