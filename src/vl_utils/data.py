import base64
import io
import json
import random
import torch
from torch.utils.data import ConcatDataset, Subset
from functools import lru_cache
from typing import Literal, Any

from PIL import Image as PILImage, ImageFile
from qwen_vl_utils import smart_resize, process_vision_info
from datasets import load_dataset, concatenate_datasets, Dataset, Image

from .common import MAX_PIXELS
from .spatial import format_point

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Dataset configurations
DATASETS = {
    "webclick": {"repo_id": "Hcompany/WebClick", "split": "test", "bbox_type": "relative"},
    "groundui-1k": {"repo_id": "agent-studio/GroundUI-1K", "split": "train", "bbox_type": "absolute"},
    "seeclick-5": {"repo_id": "andersonbcdefg/seeclick-10k-hq-annotated", "split": "train", "bbox_type": "relative"},
    "seeclick-3-4": {"repo_id": "andersonbcdefg/seeclick-10k-mid-q-annotated", "split": "train", "bbox_type": "relative"},
    "seeclick-1-2": {"repo_id": "andersonbcdefg/seeclick-10k-low-q-annotated", "split": "train", "bbox_type": "relative"},
    "seeclick-0": {"repo_id": "andersonbcdefg/seeclick-10k-awful-q-annotated", "split": "train", "bbox_type": "relative"},
    "screenspot": {"repo_id": "rootsautomation/ScreenSpot", "split": "test", "bbox_type": "relative"},
}

# LRU cache size - tune for memory vs speed tradeoff
MAX_CACHE = 16_384


def build_split_datasets(name: str, split: str = "train"):
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
        path = row["image"]["path"]
        idx  = path2idx.setdefault(path, len(path2idx))
        if idx == len(img_rows):                          # first time we see it
            img_rows.append({"id": idx, "image": row["image"]})

        # explode annotations -----------------------------------------
        if "elements" in row and row["elements"]:
            elems = row["elements"]
            if isinstance(elems, str):      # some datasets store JSON text
                elems = json.loads(elems)
            if elems and isinstance(elems[0], list):  # sometimes nested because i'm stupid
                elems = elems[0]

            for el in elems:
                ann_rows.append({
                    "img_idx": idx,
                    "instruction": el["instruction"],
                    "bbox": el["bbox"],
                })
        else:  # simple one‑annotation‑per‑row case
            ann_rows.append({
                "img_idx": idx,
                "instruction": row["instruction"],
                "bbox": row["bbox"],
            })

    images_ds = Dataset.from_list(img_rows)
    ann_ds    = Dataset.from_list(ann_rows)
    return images_ds, ann_ds

@lru_cache(maxsize=MAX_CACHE)
def _prepare_image(key, img_bytes, max_pixels: int = MAX_PIXELS):
    """Decode once, resize, stash as data‑URI."""
    with PILImage.open(io.BytesIO(img_bytes)) as im:
        im = im.convert("RGB")
        orig_w, orig_h = im.size
        # unlike  PIL, smart_resize expects h, w for some reason
        new_h, new_w  = smart_resize(orig_h, orig_w, max_pixels=max_pixels)
        im = im.resize((new_w, new_h), PILImage.Resampling.BICUBIC)

        buf = io.BytesIO()
        im.save(buf, format="JPEG", optimize=True)
        data_uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        return data_uri, (orig_w, orig_h), (new_w, new_h)

class UIAnnotationDataset(torch.utils.data.Dataset):
    """
    Uses two HF datasets:
      • images_ds[id]  -> {image: <PIL‑compatible bytes>}
      • ann_ds[i]      -> {img_idx, instruction, bbox}
    No image bytes are stored twice.
    """
    def __init__(self, images_ds, ann_ds, bbox_type="relative"):
        self.images_ds, self.ann_ds = images_ds, ann_ds
        self.bbox_type = bbox_type

    def __len__(self):
        return len(self.ann_ds)

    def __getitem__(self, idx):
        ann = self.ann_ds[idx]
        img_row = self.images_ds[ann["img_idx"]]
        img_bytes = img_row["image"]["bytes"]

        # key = (dataset‑id, img_idx) keeps cache scoped to this dataset
        key = (id(self), ann["img_idx"])
        data_uri, (orig_w, orig_h), (new_w, new_h) = _prepare_image(key, img_bytes)

        x1, y1, x2, y2 = ann["bbox"]
        if self.bbox_type == "relative":
            bb = (x1 * new_w, y1 * new_h, x2 * new_w, y2 * new_h)
        else:
            sx, sy = new_w / orig_w, new_h / orig_h
            bb = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)

        x1, y1, x2, y2  = bb
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2


        return {
            "data_uri": data_uri,
            "instruction": ann["instruction"],
            "bbox": bb,
            "click": (cx, cy),
            "orig_size": (orig_w, orig_h),
            "size": (new_w, new_h),
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
        w, h = s['size']
        scaled_bboxes.append(s['bbox'])
        label = format_point((int(cx), int(cy)), format=message_format)

        # format‑specific prompt
        if message_format == "json":
            fmt_prompt = "Report the click location as an (x, y) point in JSON."
        elif message_format == "xml":
            fmt_prompt = (
                'Report the click location in XML like '
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
    inputs: dict[str, torch.Tensor],
    IM_START: int = 151644,
    ASSISTANT: int = 77091
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
    subsample: int | None = None
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
        img_ds, ann_ds = build_split_datasets(cfg["repo_id"], cfg["split"])
        ds = UIAnnotationDataset(img_ds, ann_ds, cfg["bbox_type"])
        loaded.append(ds)

    full_dataset = loaded[0] if len(loaded) == 1 else ConcatDataset(loaded)

    if test_size == 0:
        return full_dataset, None

    # Split into train/test
    random.seed(seed)
    test_ids = set(random.sample(range(len(full_dataset)), min(test_size, len(full_dataset))))
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
    message_format: Literal["xml", "json", "plain"] = "xml"
):
    """
    Create DataLoader with the new efficient collate function.
    Uses persistent_workers to preserve LRU cache across batches.
    """
    # Create collate function with proper closure
    def collate_wrapper(batch):
        return collate_fn(
            batch,
            processor,
            eval=eval,
            message_format=message_format
        )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True,  # Critical: preserves LRU cache
        collate_fn=collate_wrapper,
        pin_memory=True,
        shuffle=not eval
    )
