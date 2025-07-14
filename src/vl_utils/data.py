import base64
import io
from os import setreuid
import random
import torch

from PIL import Image, ImageFile
from qwen_vl_utils import smart_resize
from typing import Literal, Any
from datasets import load_dataset, concatenate_datasets, Dataset
from functools import partial

from .common import MAX_PIXELS
from .spatial import format_point

from dataclasses import dataclass

@dataclass
class DatasetConfig:
    repo_id: str
    split: str
    bbox_type: Literal["absolute", "relative"]

DATASETS = {
    "webclick": DatasetConfig("Hcompany/WebClick", "test", "absolute"),
    "groundui-1k": DatasetConfig("agent-studio/GroundUI-1K", "train", "absolute"),
}

def _load_one(dataset_name: str):
    cfg = DATASETS[dataset_name]
    ds = load_dataset(cfg.repo_id, split=cfg.split).map(
        partial(convert_to_messages, bbox_type=cfg.bbox_type, format="xml"),
        batched=True,
        batch_size=32,
        num_proc=8,  # type: ignore
        load_from_cache_file=False,  # type: ignore
    ).select_columns(["messages", "bbox"])

    return ds

def load_data(
    dataset: Literal["webclick", "groundui-1k", "both"] = "webclick",
    test_size: int = 100,
    seed: int = 42
):
    if dataset == "both":
        dataset_names = ["webclick", "groundui-1k"]
    else:
        dataset_names = [dataset]

    loaded = [_load_one(ds_name) for ds_name in dataset_names]

    if len(loaded) == 1:
        ds = loaded[0]
    else:
        ds = concatenate_datasets(loaded) # type: ignore

    random.seed(seed)
    test_ids = random.sample(range(len(ds)), test_size)
    train = ds.select([x for x in range(len(ds)) if x not in test_ids])  # type: ignore
    test = ds.select(test_ids)  # type: ignore

    return train, test



def _to_data_uri(img: Image.Image, fmt="JPEG"):
    """
    Call smart_resize, then return a base-64 data URI.
    """

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    try:
        if not isinstance(img, Image.Image):
            raise TypeError(f"Unsupported type: {type(img)}")

        img = img.copy().convert("RGB")
        w, h = img.size
        new_w, new_h = smart_resize(w, h, max_pixels=MAX_PIXELS)
        img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)  # ← bicubic

        buf = io.BytesIO()
        img.save(buf, format=fmt, optimize=True)
        data = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/{fmt.lower()};base64,{data}"

    except Exception as e:
        print("⚠️  to_data_uri failed:", repr(e))
        return None


def convert_to_messages(batch: dict, bbox_type="relative", format: Literal["plain", "xml", "json"] = "plain"):
    out, bboxes = [], []

    for img, inst, (x1, y1, x2, y2) in zip(
        batch["image"], batch["instruction"], batch["bbox"]
    ):
        width, height = img.size
        new_w, new_h = smart_resize(width, height, max_pixels=MAX_PIXELS)
        data_uri = _to_data_uri(img)
        if data_uri is None:
            continue
        if bbox_type == "relative":
            cx, cy = (x1 + x2) / 2 * new_w, (y1 + y2) / 2 * new_h
        else:
            # scale by the scale factor
            w_scale = new_w / width
            h_scale = new_h / height
            cx, cy = (x1 + x2) / 2 * w_scale, (y1 + y2) / 2 * h_scale

        cx, cy = int(cx), int(cy)
        new_w, new_h = int(new_w), int(new_h)

        label = format_point((cx, cy), format=format)
        format_prompt = "Return your click as (x, y) pixel coordinates."
        if format == "json":
            format_prompt = "Report the click location as an (x, y) point in JSON format."
        elif format == "xml":
            format_prompt = 'Report the click location in XML like <points x1="x" y1="y">object</points>.'

        out.append(
            [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Determine where to click in the UI to complete the task. "
                                + format_prompt
                                + f" The image is size [{new_w}, {new_h}], "
                                + f" (0, 0) is the top-left and ({new_w}, {new_h}) is the bottom-right."
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": data_uri},
                        {"type": "text", "text": inst},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": label}]},
            ]
        )
        if bbox_type == "relative":
            bboxes.append((x1 * new_w, y1 * new_h, x2 * new_w, y2 * new_h))
        else:
            bboxes.append((x1 * w_scale, y1 * h_scale, x2 * w_scale, y2 * h_scale))  # type: ignore

    return {"messages": out, "bbox": bboxes}


def strip_null_images(convs):
    for turn in convs:
        for part in turn["content"]:
            if part.get("image") is None:
                part.pop("image")  # delete the bogus key
            if "video" in part and part.get("video") is None:
                part.pop("video")
    return convs

def add_labels_to_inputs(
    inputs: dict[str, torch.Tensor], IM_START = 151644, ASSISTANT=77091
) -> dict[str, torch.Tensor]:
    labels = inputs["input_ids"].clone()

    for i, ids in enumerate(inputs["input_ids"]):
        # locate the last "<|im_start|> assistant" pair – that’s the answer we want
        starts = (ids == IM_START).nonzero(as_tuple=True)[0]
        tgt_start = None
        for s in reversed(starts.tolist()):
            if s + 1 < len(ids) and ids[s + 1] == ASSISTANT:
                tgt_start = s + 2  # first token after “assistant”
                break
        if tgt_start is None:
            raise ValueError("no assistant response found in input")
        else:
            labels[i, :tgt_start] = -100  # ignore system/user & image tokens

    inputs["labels"] = labels

    return inputs

def _strip_answer(conv: list[dict]) -> list[dict]:
    """Remove the final assistant message so the model must predict it."""
    if conv and conv[-1]["role"] == "assistant":
        return conv[:-1]
    return conv

def collate(batch: list[dict[str, Any]], processor: Any, eval: bool = False):
    conversations = [strip_null_images(item["messages"]) for item in batch]
    if eval:
        conversations = [_strip_answer(conv) for conv in conversations]

        texts = [
            processor.apply_chat_template(c, add_generation_prompt=eval, tokenize=False)
            for c in conversations
        ]

        image_inputs, _ = process_vision_info(conversations)  # type: ignore

        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt",
        )

        inputs = add_labels_to_inputs(inputs)
        if eval:
            inputs["bbox"] = [item['bbox'] for item in batch] # type: ignore
        return inputs
