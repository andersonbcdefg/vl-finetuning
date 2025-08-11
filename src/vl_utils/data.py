import base64
import io
import json
import os
import random
from datetime import datetime
from typing import Any, Literal, cast

import datasets
import torch
from datasets import Dataset, Image, load_dataset
from PIL import Image as PILImage
from PIL import ImageFile
from qwen_vl_utils import process_vision_info, smart_resize
from torch.utils.data import ConcatDataset, Subset

from .common import MAX_PIXELS
from .spatial import bbox_center, flat_polygon_to_bbox, format_point, scale_bbox

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Dataset configurations
DATASETS = {
    "webclick": {
        "repo_id": "Hcompany/WebClick",
        "split": "test",
        "bbox_type": "relative",
        "bbox_format": "xyxy",
    },
    "groundui-1k": {
        "repo_id": "agent-studio/GroundUI-1K",
        "split": "train",
        "bbox_type": "absolute",
        "bbox_format": "xyxy",
    },
    "seeclick-5": {
        "repo_id": "andersonbcdefg/seeclick-10k-hq-annotated",
        "split": "train",
        "bbox_type": "relative",
        "bbox_format": "xyxy",
    },
    "seeclick-3-4": {
        "repo_id": "andersonbcdefg/seeclick-10k-mid-q-annotated",
        "split": "train",
        "bbox_type": "relative",
        "bbox_format": "xyxy",
    },
    "seeclick-1-2": {
        "repo_id": "andersonbcdefg/seeclick-10k-low-q-annotated",
        "split": "train",
        "bbox_type": "relative",
        "bbox_format": "xyxy",
    },
    "seeclick-0": {
        "repo_id": "andersonbcdefg/seeclick-10k-awful-q-annotated",
        "split": "train",
        "bbox_type": "relative",
        "bbox_format": "xyxy",
    },
    "seeclick-0-annots": {
        "repo_id": "andersonbcdefg/seeclick-0-1-yolo-annots",
        "split": "train",
        "bbox_type": "relative",
        "bbox_format": "xyxy",
    },
    "seeclick-1-annots": {
        "repo_id": "andersonbcdefg/seeclick-2-3-yolo-annots",
        "split": "train",
        "bbox_type": "relative",
        "bbox_format": "xyxy",
    },
    "seeclick-2-annots": {
        "repo_id": "andersonbcdefg/seeclick-4-7-yolo-annots",
        "split": "train",
        "bbox_type": "relative",
        "bbox_format": "xyxy",
    },
    "seeclick-3-annots": {
        "repo_id": "andersonbcdefg/seeclick-8-12-yolo-annots",
        "split": "train",
        "bbox_type": "relative",
        "bbox_format": "xyxy",
    },
    "seeclick-4-annots": {
        "repo_id": "andersonbcdefg/seeclick-13-plus-yolo-annots",
        "split": "train",
        "bbox_type": "relative",
        "bbox_format": "xyxy",
    },
    "screenspot": {
        "repo_id": "rootsautomation/ScreenSpot",
        "split": "test",
        "bbox_type": "relative",
        "bbox_format": "xyxy",
    },
    "screenspot-v2-web": {
        "images_repo": "andersonbcdefg/screenspot-v2-images",
        "images_split": "train",
        "ann_repo": "andersonbcdefg/screenspot-v2-annots-web",
        "ann_split": "train",
        "bbox_type": "absolute",
        "bbox_format": "xywh",
    },
    "screenspot-v2-mobile": {
        "images_repo": "andersonbcdefg/screenspot-v2-images",
        "images_split": "train",
        "ann_repo": "andersonbcdefg/screenspot-v2-annots-mobile",
        "ann_split": "train",
        "bbox_type": "absolute",
        "bbox_format": "xywh",
    },
    "screenspot-v2-desktop": {
        "images_repo": "andersonbcdefg/screenspot-v2-images",
        "images_split": "train",
        "ann_repo": "andersonbcdefg/screenspot-v2-annots-desktop",
        "ann_split": "train",
        "bbox_type": "absolute",
        "bbox_format": "xywh",
    },
    # uses original instruction from osworld
    "osworld-g-orig": {
        "images_repo": "andersonbcdefg/osworld-g-images",
        "images_split": "train",
        "ann_repo": "andersonbcdefg/osworld-g-orig-annots",
        "ann_split": "train",
        "bbox_type": "absolute",
        "bbox_format": "xyxy",
    },
    # uses "refined" instruction meant to not require add'l context
    "osworld-g-refined": {
        "images_repo": "andersonbcdefg/osworld-g-images",
        "images_split": "train",
        "ann_repo": "andersonbcdefg/osworld-g-refined-annots",
        "ann_split": "train",
        "bbox_type": "absolute",
        "bbox_format": "xyxy",
    },
    # TRAINING VARIATIONS
    "seeclick-filtered": {
        "images_repo": "andersonbcdefg/seeclick-10k-scored",
        "images_split": "train",
        "ann_repo": "andersonbcdefg/seeclick-filtered-orig-labels",
        "ann_split": "train",
        "bbox_type": "relative",
        "bbox_format": "xyxy",
    },
    "seeclick-relabeled": {
        "images_repo": "andersonbcdefg/seeclick-10k-scored",
        "images_split": "train",
        "ann_repo": "andersonbcdefg/seeclick-filtered-relabeled",
        "ann_split": "train",
        "bbox_type": "relative",
        "bbox_format": "xyxy",
    },
    "seeclick-yolo-labeled": {
        "images_repo": "andersonbcdefg/seeclick-10k-scored",
        "images_split": "train",
        "ann_repo": "andersonbcdefg/seeclick-filtered-yolo-labeled",
        "ann_split": "train",
        "bbox_type": "absolute",
        "bbox_format": "xyxy",
    },
    "seeclick-gemini-labeled": {
        "images_repo": "andersonbcdefg/seeclick-10k-scored",
        "images_split": "train",
        "ann_repo": "andersonbcdefg/seeclick-filtered-gem-labeled",
        "ann_split": "train",
        "bbox_type": "relative",
        "bbox_format": "xyxy",
    },
    "vibe-coded": {
        "images_repo": "andersonbcdefg/vibe-coded",
        "images_split": "train",
        "ann_repo": "andersonbcdefg/vibe-coded-labeled",
        "ann_split": "train",
        "bbox_type": "relative",
        "bbox_format": "xyxy",
    },
    "vibe-coded-filtered": {
        "images_repo": "andersonbcdefg/vibe-coded",
        "images_split": "train",
        "ann_repo": "andersonbcdefg/vibe-coded-gem-labeled-filtered",
        "ann_split": "train",
        "bbox_type": "relative",
        "bbox_format": "xyxy",
    },
    "calendars": {
        "images_repo": "andersonbcdefg/synth-calendars",
        "images_split": "train",
        "ann_repo": "andersonbcdefg/synth-calendar-annots",
        "ann_split": "train",
        "bbox_type": "absolute",
        "bbox_format": "xyxy",
    },
}


def _prepare_image(img_bytes, max_pixels: int = MAX_PIXELS):
    """Decode once, resize, stash as data‑URI."""
    with PILImage.open(io.BytesIO(img_bytes)) as im:
        im = im.convert("RGB")
        orig_w, orig_h = im.size
        new_h, new_w = smart_resize(orig_h, orig_w, max_pixels=max_pixels)
        im = im.resize((new_w, new_h), PILImage.Resampling.BICUBIC)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", optimize=True)
        data_uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        return data_uri, (orig_w, orig_h), (new_w, new_h)


def build_split_datasets(
    name: str,
    split: str,
    bbox_type: Literal["relative", "absolute"],
    bbox_format: Literal["xyxy", "xywh"] = "xyxy",
):
    """
    Returns:
      images_ds : Dataset(id, image)          # one row per unique image
      ann_ds    : Dataset(img_idx, instr, bb) # one row per annotation
    The image column is *not decoded* here, so memory stays low.
    """
    raw = load_dataset(name, split=split).cast_column("image", Image(decode=False))

    path2idx, img_rows, ann_rows = {}, [], []
    for row in raw:
        # identify the image
        orig_bytes = path = row["image"]["bytes"]  # type: ignore
        data_uri, orig_size, new_size = _prepare_image(orig_bytes)
        idx = path2idx.setdefault(path, len(path2idx))
        if idx == len(img_rows):  # first time we see it
            img_rows.append({"id": idx, "uri": data_uri})  # type: ignore

        # explode annotations -----------------------------------------
        row = cast(dict, row)
        if "elements" in row and row["elements"]:  # type: ignore
            elems = row["elements"]  # type: ignore
            if isinstance(elems, str):  # some datasets store JSON text
                elems = json.loads(elems)
            if elems and isinstance(
                elems[0], list
            ):  # sometimes nested because i'm stupid
                elems = elems[0]

            for el in elems:
                bbox = el["bbox"]
                if len(bbox) > 4:
                    bbox = flat_polygon_to_bbox(bbox)
                # 0, 0, 0, 0 is usually refusal - we should do these later
                if all(x <= 0 for x in bbox):
                    continue
                scaled = scale_bbox(bbox, bbox_type, bbox_format, orig_size, new_size)
                if scaled[0] > scaled[2] or scaled[1] > scaled[3]:
                    continue
                ann_rows.append(
                    {
                        "img_idx": idx,
                        "instruction": el["instruction"],
                        "bbox": scaled,
                        "center": bbox_center(scaled),  # type: ignore
                        "size": new_size,
                    }
                )
        elif "seeclick_elements" in row and row["seeclick_elements"]:
            elems = row["seeclick_elements"]
            if isinstance(elems, str):  # some datasets store JSON text
                elems = json.loads(elems)
            if elems and isinstance(
                elems[0], list
            ):  # sometimes nested because i'm stupid
                elems = elems[0]

            for el in elems:
                bbox = el["bbox"]
                if len(bbox) > 4:
                    bbox = flat_polygon_to_bbox(bbox)
                if all(x <= 0 for x in bbox):
                    continue
                scaled = scale_bbox(bbox, bbox_type, bbox_format, orig_size, new_size)
                if scaled[0] > scaled[2] or scaled[1] > scaled[3]:
                    continue
                ann_rows.append(
                    {
                        "img_idx": idx,
                        "instruction": el["instruction"],
                        "bbox": scaled,
                        "center": bbox_center(scaled),  # type: ignore
                        "size": new_size,
                    }
                )
        else:  # simple one‑annotation‑per‑row case
            bbox = row["bbox"]
            if len(bbox) > 4:
                bbox = flat_polygon_to_bbox(bbox)
            if all(x <= 0 for x in bbox):
                continue
            scaled = scale_bbox(bbox, bbox_type, bbox_format, orig_size, new_size)  # type: ignore
            if scaled[0] > scaled[2] or scaled[1] > scaled[3]:
                pass
            else:
                ann_rows.append(
                    {
                        "img_idx": idx,
                        "instruction": row["instruction"],  # type: ignore
                        "bbox": scaled,
                        "center": bbox_center(scaled),  # type: ignore
                        "size": new_size,
                    }
                )

    images_ds = Dataset.from_list(img_rows)
    ann_ds = Dataset.from_list(ann_rows)
    return images_ds, ann_ds


def build_split_datasets_two_sources(
    image_repo: str,
    image_split: str,
    ann_repo: str,
    ann_split: str,
    bbox_type: Literal["relative", "absolute"],
    bbox_format: Literal["xyxy", "xywh"] = "xyxy",
):
    """Join images from one dataset with annotations from another."""

    def _time():
        # get time formatted as [HH:MM:SS]
        return datetime.now().strftime("[%H:%M:%S]")

    print(f"{_time()} loading image dataset: {image_repo}")
    images_raw = load_dataset(image_repo, split=image_split).cast_column(
        "image", Image(decode=False)
    )

    print(f"{_time()} loading annotation dataset: {ann_repo}")
    ann_raw = load_dataset(ann_repo, split=ann_split, download_mode="force_redownload")
    images_raw = cast(datasets.Dataset, images_raw)
    ann_raw = cast(datasets.Dataset, ann_raw)
    path_map: dict[str, dict] = {}
    img_rows, ann_rows = [], []

    # normalize filenames, skip images that don't have annotations
    def _normalize_filename(row):
        fname = (
            row.get("file_name")
            or row.get("img_filename")
            or row.get("filename")
            or row.get("path")
        )
        if not fname:
            raise ValueError("Image filename not found")

        return {"img_filename": os.path.basename(fname)}

    ann_raw = ann_raw.map(_normalize_filename)
    present_filenames = set(list(ann_raw["img_filename"]))

    print(f"{_time()} preparing images...")

    def _prepare_fn(row, idx):
        path = row["image"]["path"]  # type: ignore
        data_uri, orig_size, new_size = _prepare_image(row["image"]["bytes"])  # type: ignore

        return {
            "id": idx,
            "uri": data_uri,
            "path": os.path.basename(path),
            "orig_size": orig_size,
            "new_size": new_size,
        }

    images_raw = (
        images_raw.cast_column("image", datasets.Image(decode=False))
        .filter(lambda row: os.path.basename(row["image"]["path"]) in present_filenames)
        .map(_prepare_fn, with_indices=True, remove_columns=["image"], num_proc=4)
    )

    for row in images_raw:
        row = cast(dict, row)
        idx = row["id"]
        data_uri = row["uri"]
        path = row["path"]
        orig_size = row["orig_size"]
        new_size = row["new_size"]
        img_rows.append({"id": idx, "uri": data_uri})
        info = {"idx": idx, "orig": orig_size, "new": new_size}
        path_map[path] = info

    print(f"{_time()} preparing annotations...")
    for row in ann_raw:
        row = cast(dict, row)
        info = path_map.get(row["img_filename"])
        if info is None:
            continue
        orig_size, new_size = info["orig"], info["new"]
        idx = info["idx"]

        if "elements" in row and row["elements"]:
            elems = row["elements"]
            if isinstance(elems, str):
                elems = json.loads(elems)
            if elems and isinstance(elems[0], list):
                elems = elems[0]
            for el in elems:
                bbox = el["bbox"]
                if len(bbox) > 4:
                    bbox = flat_polygon_to_bbox(bbox)
                if all(x <= 0 for x in bbox):
                    continue
                scaled = scale_bbox(bbox, bbox_type, bbox_format, orig_size, new_size)
                if scaled[0] > scaled[2] or scaled[1] > scaled[3]:
                    continue
                ann_rows.append(
                    {
                        "img_idx": idx,
                        "instruction": el["instruction"],
                        "bbox": scaled,
                        "center": bbox_center(scaled),
                        "size": new_size,
                    }
                )
        else:
            if "bbox" not in row or "instruction" not in row:
                continue
            bbox = row["bbox"]
            if len(bbox) > 4:
                bbox = flat_polygon_to_bbox(bbox)
            if all(x <= 0 for x in bbox):
                continue
            scaled = scale_bbox(bbox, bbox_type, bbox_format, orig_size, new_size)
            if scaled[0] > scaled[2] or scaled[1] > scaled[3]:
                continue

            ann_rows.append(
                {
                    "img_idx": idx,
                    "instruction": row["instruction"],
                    "bbox": scaled,
                    "center": bbox_center(scaled),
                    "size": new_size,
                }
            )

    print(f"{_time()} creating datasets...")
    images_ds = Dataset.from_list(img_rows)
    ann_ds = Dataset.from_list(ann_rows)

    print(f"{_time()} done.")
    return images_ds, ann_ds


class UIAnnotationDataset(torch.utils.data.Dataset):
    """
    Uses two HF datasets:
      • images_ds[id]  -> {image: <PIL‑compatible bytes>}
      • ann_ds[i]      -> {img_idx, instruction, bbox}
    No image bytes are stored twice.
    """

    def __init__(self, images_ds, ann_ds, bbox_type="relative"):
        self.images_ds, self.ann_ds = images_ds, ann_ds

    def __len__(self):
        return len(self.ann_ds)

    def __getitem__(self, idx):
        ann = self.ann_ds[idx]
        img_row = self.images_ds[ann["img_idx"]]

        return {
            "data_uri": img_row["uri"],
            "instruction": ann["instruction"],
            "bbox": ann["bbox"],
            "click": ann["center"],
            "size": ann["size"],
        }


# ---- collate_fn --------------------------------------------------
def collate_fn(
    batch: list[dict[str, Any]],
    processor,
    *,
    eval: bool = False,
    message_format: Literal["xml", "plain", "json"] = "xml",
):
    """Build chat prompts + tensors from UIAnnotationDataset samples."""
    messages, scaled_bboxes = [], []

    for s in batch:
        data_uri = s["data_uri"]
        cx, cy = s["click"]
        w, h = s["size"]
        scaled_bboxes.append(s["bbox"])
        label = format_point((int(cx), int(cy)), format=message_format)

        # format‑specific prompt
        if message_format == "json":
            fmt_prompt = "Report the click location as an (x, y) point in JSON."
        elif message_format == "xml":
            fmt_prompt = (
                "Report the click location in XML like "
                '<points x1="x" y1="y">object</points>.'
            )
        else:  # plain
            fmt_prompt = "Return your click as (x, y) pixel coordinates."

        conversation = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Determine where to click in the UI to complete the instruction/task. {fmt_prompt} "
                            + f"Image size [{w}, {h}],"
                            + f" (0, 0) is the top-left and ({w}, {h}) is the bottom-right."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": data_uri},
                    {"type": "text", "text": s["instruction"]},
                ],
            },
        ]

        if not eval:  # training—append target
            conversation.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": label}],
                }
            )

        messages.append(conversation)

    # ---- tokenizer / processor -----------------------------------
    texts = [
        processor.apply_chat_template(m, add_generation_prompt=eval, tokenize=False)
        for m in messages
    ]
    img_inputs, _ = process_vision_info(messages)  # type: ignore

    out = processor(text=texts, images=img_inputs, padding=True, return_tensors="pt")

    if not eval:
        out = add_labels_to_inputs(out)
    else:
        out["bbox"] = scaled_bboxes

    return out


def add_labels_to_inputs(
    inputs: dict[str, torch.Tensor], IM_START: int = 151644, ASSISTANT: int = 77091
) -> dict[str, torch.Tensor]:
    """Add labels, masking everything except assistant response."""
    labels = inputs["input_ids"].clone()

    for i, ids in enumerate(inputs["input_ids"]):
        # Locate the last "<|im_start|> assistant" pair
        starts = (ids == IM_START).nonzero(as_tuple=True)[0]
        tgt_start = None
        for s in reversed(starts.tolist()):
            if s + 1 < len(ids) and ids[s + 1] == ASSISTANT:
                tgt_start = s + 2  # first token after "assistant"
                break
        if tgt_start is None:
            raise ValueError("no assistant response found in input")
        else:
            labels[i, :tgt_start] = -100  # ignore system/user & image tokens

    inputs["labels"] = labels
    return inputs


def load_data(
    datasets: str | list[str] = "webclick",
    test_size: int = 100,
    seed: int = 42,
    subsample: int | None = None,
):
    """
    Load datasets using the new efficient approach.

    Returns train and test UIAnnotationDataset instances.
    Backwards compatible with the old load_data interface.
    """
    if isinstance(datasets, str):
        dataset_names = [datasets]
    else:
        dataset_names = datasets

    # Validate dataset names
    for dsn in dataset_names:
        assert dsn in DATASETS, f"dataset {dsn} not found"

    # Load lightweight datasets
    loaded = []
    for dsn in dataset_names:
        cfg = DATASETS[dsn]
        if "repo_id" in cfg:
            img_ds, ann_ds = build_split_datasets(
                cfg["repo_id"],
                cfg["split"],
                cfg["bbox_type"],  # type: ignore
                cfg.get("bbox_format", "xyxy"),  # type: ignore
            )
        else:
            img_ds, ann_ds = build_split_datasets_two_sources(
                cfg["images_repo"],
                cfg.get("images_split", "train"),
                cfg["ann_repo"],
                cfg.get("ann_split", "train"),
                cfg["bbox_type"],  # type: ignore
                cfg.get("bbox_format", "xyxy"),  # type: ignore
            )
        ds = UIAnnotationDataset(img_ds, ann_ds)
        loaded.append(ds)

    full_dataset = loaded[0] if len(loaded) == 1 else ConcatDataset(loaded)

    if test_size == 0:
        return full_dataset, None

    # Split into train/test
    random.seed(seed)
    test_ids = set(
        random.sample(range(len(full_dataset)), min(test_size, len(full_dataset)))
    )
    train_ids = [i for i in range(len(full_dataset)) if i not in test_ids]

    train_dataset = Subset(full_dataset, train_ids)
    test_dataset = Subset(full_dataset, list(test_ids))

    return train_dataset, test_dataset


def create_dataloader(
    dataset,
    processor,
    batch_size: int = 32,
    num_workers: int = 4,
    eval: bool = False,
    message_format: Literal["xml", "json", "plain"] = "xml",
):
    """
    Create DataLoader with the new efficient collate function.
    Uses persistent_workers to preserve LRU cache across batches.
    """

    # Create collate function with proper closure
    def collate_wrapper(batch):
        return collate_fn(batch, processor, eval=eval, message_format=message_format)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True,  # Critical: preserves LRU cache
        collate_fn=collate_wrapper,
        pin_memory=True,
        shuffle=not eval,
    )
