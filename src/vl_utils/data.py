import base64
import io
import json
import random
import torch
from functools import lru_cache
from typing import Literal, Any

from PIL import Image, ImageFile
from qwen_vl_utils import smart_resize, process_vision_info
from datasets import load_dataset, concatenate_datasets, Dataset

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
    "seeclick-0": {"repo_id": "andersonbcdefg/seeclick-10k-low-q-annotated", "split": "train", "bbox_type": "relative"},
    "screenspot": {"repo_id": "rootsautomation/ScreenSpot", "split": "test", "bbox_type": "relative"},
}

# LRU cache size - tune for memory vs speed tradeoff
MAX_CACHE = 16_384


def lightweight_hf_ds(name: str, split: str = "train"):
    """Load HF dataset but keep only lightweight data (paths, instructions, bboxes)."""
    ds = load_dataset(name, split=split)

    def drop_bytes(batch):
        # Extract just the path from image objects, don't keep PIL data
        paths = []
        for img in batch["image"]:
            if isinstance(img, dict) and "path" in img:
                paths.append(img["path"])
            elif hasattr(img, "filename"):
                paths.append(img.filename)
            else:
                # For datasets where image is already a path or needs conversion
                paths.append(str(img))
        return {"path": paths}

    # Determine which columns to keep based on what's available
    available_cols = set(ds.column_names)
    cols_to_keep = ["path"]

    if "instruction" in available_cols:
        cols_to_keep.append("instruction")
    if "bbox" in available_cols:
        cols_to_keep.append("bbox")
    if "elements" in available_cols:
        cols_to_keep.append("elements")

    ds = ds.map(drop_bytes, batched=True)
    ds = ds.remove_columns(set(ds.column_names) - set(cols_to_keep))

    return ds


@lru_cache(maxsize=MAX_CACHE)
def _prepare_image(path: str, max_pixels: int = MAX_PIXELS):
    """Resize once & keep the data-URI in process memory."""
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            orig_w, orig_h = im.size

            # Smart resize
            new_w, new_h = smart_resize(orig_w, orig_h, max_pixels=max_pixels)
            im_resized = im.resize((new_w, new_h), Image.Resampling.BICUBIC)

            # Convert to data URI
            buf = io.BytesIO()
            im_resized.save(buf, format="JPEG", optimize=True)
            data = base64.b64encode(buf.getvalue()).decode()
            data_uri = f"data:image/jpeg;base64,{data}"

            return data_uri, orig_w, orig_h, new_w, new_h
    except Exception as e:
        print(f"⚠️  _prepare_image failed for {path}: {repr(e)}")
        raise e


class UIAnnotationDataset(torch.utils.data.Dataset):
    """
    Two-level dataset structure:
    - One list of unique images (paths)
    - One list of annotations that points into the image list
    """
    def __init__(self, hf_ds, bbox_type="relative", max_pixels=MAX_PIXELS):
        # Unique images
        self.paths: list[str] = []
        path_to_idx: dict[str, int] = {}

        # Flat list of (img_idx, instruction, bbox)
        self.ann: list[tuple[int, str, tuple[float, float, float, float]]] = []

        # Handle different dataset formats
        if "elements" in hf_ds.column_names:
            # Format with elements column (multiple annotations per image)
            for path, elements in zip(hf_ds["path"], hf_ds["elements"]):
                # Get or create image index
                i = path_to_idx.setdefault(path, len(self.paths))
                if i == len(self.paths):
                    self.paths.append(path)

                # Parse elements
                if isinstance(elements, str):
                    elements = json.loads(elements)
                if isinstance(elements, list) and len(elements) > 0:
                    elements = elements[0]  # Take first element group

                for elem in elements:
                    inst = elem["instruction"]
                    bbox = elem["bbox"]
                    self.ann.append((i, inst, tuple(bbox)))
        else:
            # Simple format (one annotation per row)
            for path, inst, bb in zip(hf_ds["path"], hf_ds["instruction"], hf_ds["bbox"]):
                i = path_to_idx.setdefault(path, len(self.paths))
                if i == len(self.paths):
                    self.paths.append(path)
                self.ann.append((i, inst, tuple(bb)))

        self.bbox_type = bbox_type
        self.max_pixels = max_pixels

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, idx):
        img_idx, inst, bb = self.ann[idx]
        return {"img_idx": img_idx, "instruction": inst, "bbox": bb}


def collate_fn(
    batch,
    idx_to_path,
    processor,
    bbox_type="relative",
    eval=False,
    message_format: Literal["xml", "plain", "json"]="xml"
):
    """Collate function that builds messages on the fly."""
    messages, scaled_bboxes = [], []

    for sample in batch:
        # Get processed image data
        data_uri, W, H, w, h = _prepare_image(idx_to_path[sample["img_idx"]])
        x1, y1, x2, y2 = sample["bbox"]

        # Scale bbox coordinates
        if bbox_type == "relative":
            # bbox is already 0-1, scale to new image size
            cx, cy = (x1 + x2) / 2 * w, (y1 + y2) / 2 * h
            bb = (x1 * w, y1 * h, x2 * w, y2 * h)
        else:
            # bbox is absolute, scale by resize factor
            sx, sy = w / W, h / H
            cx, cy = (x1 + x2) / 2 * sx, (y1 + y2) / 2 * sy
            bb = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)

        # Format the target point
        label = format_point((int(cx), int(cy)), format=message_format)

        # Create format-specific prompt
        if message_format == "json":
            format_prompt = "Report the click location as an (x, y) point in JSON format."
        elif message_format == "xml":
            format_prompt = 'Report the click location in XML like <points x1="x" y1="y">object</points>.'
        else:
            format_prompt = "Return your click as (x, y) pixel coordinates."

        # Build conversation
        conversation = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Determine where to click in the UI. "
                            f"{format_prompt} Image size [{w}, {h}]."
                        )
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": data_uri},
                    {"type": "text", "text": sample["instruction"]},
                ],
            },
        ]

        # Add assistant response for training
        if not eval:
            conversation.append({
                "role": "assistant",
                "content": [{"type": "text", "text": label}]
            })

        messages.append(conversation)
        scaled_bboxes.append(bb)

    # Process with Qwen
    texts = [processor.apply_chat_template(m, add_generation_prompt=eval, tokenize=False)
             for m in messages]
    img_inputs, _ = process_vision_info(messages) # type: ignore

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
    loaded_ds = []
    for dsn in dataset_names:
        cfg = DATASETS[dsn]
        hf_ds = lightweight_hf_ds(cfg["repo_id"], cfg["split"])
        loaded_ds.append((hf_ds, cfg["bbox_type"]))

    # Combine datasets if multiple
    if len(loaded_ds) == 1:
        combined_hf_ds, bbox_type = loaded_ds[0]
    else:
        # Concatenate HF datasets
        hf_datasets = [ds for ds, _ in loaded_ds]
        combined_hf_ds = concatenate_datasets(hf_datasets)
        bbox_type = loaded_ds[0][1]  # Use first dataset's bbox_type

    # Create UIAnnotationDataset
    full_dataset = UIAnnotationDataset(combined_hf_ds, bbox_type=bbox_type)


    if test_size == 0:
        return full_dataset, None

    # Split into train/test
    random.seed(seed)
    test_ids = set(random.sample(range(len(full_dataset)), min(test_size, len(full_dataset))))
    train_ids = [i for i in range(len(full_dataset)) if i not in test_ids]

    train_dataset = torch.utils.data.Subset(full_dataset, train_ids)
    test_dataset = torch.utils.data.Subset(full_dataset, list(test_ids))

    return train_dataset, test_dataset


def create_dataloader(
    dataset,
    processor,
    batch_size: int = 32,
    num_workers: int = 4,
    eval: bool = False,
    message_format: str = "xml"
):
    """
    Create DataLoader with the new efficient collate function.
    Uses persistent_workers to preserve LRU cache across batches.
    """
    # Get the underlying UIAnnotationDataset
    if isinstance(dataset, torch.utils.data.Subset):
        base_dataset = dataset.dataset
    else:
        base_dataset = dataset

    # Create collate function with proper closure
    def collate_wrapper(batch):
        return collate_fn(
            batch,
            base_dataset.paths,
            processor,
            bbox_type=base_dataset.bbox_type,
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
