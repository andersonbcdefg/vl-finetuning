import base64
import io
import json
import random
import torch
import bisect
from PIL import Image, ImageFile
from qwen_vl_utils import smart_resize, process_vision_info
from typing import Literal, Any, Optional
from datasets import load_dataset, Dataset
from functools import lru_cache
from dataclasses import dataclass
from .common import MAX_PIXELS
from .spatial import format_point

ImageFile.LOAD_TRUNCATED_IMAGES = True

@dataclass
class DatasetConfig:
    repo_id: str
    split: str
    bbox_type: Literal["absolute", "relative"]

DATASETS = {
    "webclick": DatasetConfig("Hcompany/WebClick", "test", "relative"),
    "groundui-1k": DatasetConfig("agent-studio/GroundUI-1K", "train", "absolute"),
    "seeclick-5": DatasetConfig("andersonbcdefg/seeclick-10k-hq-annotated", "train", "relative"),
    "seeclick-3-4": DatasetConfig("andersonbcdefg/seeclick-10k-mid-q-annotated", "train", "relative"),
    "seeclick-1-2": DatasetConfig("andersonbcdefg/seeclick-10k-low-q-annotated", "train", "relative"),
    "seeclick-0": DatasetConfig("andersonbcdefg/seeclick-10k-low-q-annotated", "train", "relative"),
}


class ProcessedImage:
    """Holds a processed image with its data URI and dimensions."""
    def __init__(self, data_uri: str, width: int, height: int, orig_width: int, orig_height: int):
        self.data_uri = data_uri
        self.width = width
        self.height = height
        self.orig_width = orig_width
        self.orig_height = orig_height
        self.scale_x = width / orig_width
        self.scale_y = height / orig_height


class ImageAnnotationDataset(torch.utils.data.Dataset):
    """
    A clean dataset that:
    1. Stores raw data (image paths/bytes, instructions, bboxes)
    2. Processes images once on first access and caches them
    3. Defers message creation to the collate function
    """
    def __init__(
        self,
        dataset_name: str,
        images: list[Any],  # Can be PIL Images or paths
        instructions: list[list[str]],
        bboxes: list[list[tuple[float, float, float, float]]],
        bbox_type: Literal["absolute", "relative"] = "relative",
        max_pixels: int = MAX_PIXELS
    ):
        self.dataset_name = dataset_name
        self.images = images
        self.instructions = instructions
        self.bboxes = bboxes
        self.bbox_type = bbox_type
        self.max_pixels = max_pixels

        # Cache for processed images
        self._processed_cache = {}

        # Flatten the dataset structure for easy indexing
        self.flat_data = []
        for img_idx, (img, inst_list, bbox_list) in enumerate(zip(images, instructions, bboxes)):
            for inst, bbox in zip(inst_list, bbox_list):
                self.flat_data.append({
                    'img_idx': img_idx,
                    'instruction': inst,
                    'bbox': bbox
                })

    def __len__(self) -> int:
        return len(self.flat_data)

    def _process_image(self, img_idx: int) -> ProcessedImage:
        """Process an image once and cache the result."""
        if img_idx in self._processed_cache:
            return self._processed_cache[img_idx]

        img = self.images[img_idx]
        if not isinstance(img, Image.Image):
            img = Image.open(img)

        img = img.convert("RGB")
        orig_w, orig_h = img.size

        # Smart resize
        new_w, new_h = smart_resize(orig_w, orig_h, max_pixels=self.max_pixels)
        img_resized = img.resize((new_w, new_h), Image.Resampling.BICUBIC)

        # Convert to data URI
        buf = io.BytesIO()
        img_resized.save(buf, format="JPEG", optimize=True)
        data = base64.b64encode(buf.getvalue()).decode()
        data_uri = f"data:image/jpeg;base64,{data}"

        processed = ProcessedImage(data_uri, new_w, new_h, orig_w, orig_h)
        self._processed_cache[img_idx] = processed
        return processed

    def __getitem__(self, idx: int):
        item = self.flat_data[idx]
        processed_img = self._process_image(item['img_idx'])

        # Transform bbox coordinates based on bbox_type
        x1, y1, x2, y2 = item['bbox']
        if self.bbox_type == "relative":
            # Convert relative coords to absolute in resized image
            x1, x2 = x1 * processed_img.width, x2 * processed_img.width
            y1, y2 = y1 * processed_img.height, y2 * processed_img.height
        else:
            # Scale absolute coords to resized image
            x1, x2 = x1 * processed_img.scale_x, x2 * processed_img.scale_x
            y1, y2 = y1 * processed_img.scale_y, y2 * processed_img.scale_y

        return {
            'processed_image': processed_img,
            'instruction': item['instruction'],
            'bbox': (x1, y1, x2, y2),
            'dataset_name': self.dataset_name
        }


def load_datasets(
    dataset_names: str | list[str] = "webclick",
    test_size: int = 100,
    seed: int = 42,
) -> tuple[ImageAnnotationDataset, ImageAnnotationDataset]:
    """Load and combine multiple datasets."""
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]

    all_images = []
    all_instructions = []
    all_bboxes = []

    for dsn in dataset_names:
        assert dsn in DATASETS, f"Dataset {dsn} not found"
        cfg = DATASETS[dsn]

        # Load raw dataset
        ds: Dataset = load_dataset(cfg.repo_id, split=cfg.split) # type: ignore

        # Handle different dataset formats
        if "elements" in ds.column_names:
            # Parse elements column
            def parse_elements(example):
                elements = json.loads(example['elements'][0]) if isinstance(example['elements'], str) else example['elements']
                instructions = []
                bboxes = []
                for elem in elements:
                    instructions.append(elem['instruction'])
                    bbox = elem['bbox']
                    bboxes.append((bbox[0], bbox[1], bbox[2], bbox[3]))
                return {
                    'image': example['image'],
                    'instructions': instructions,
                    'bboxes': bboxes
                }

            processed = ds.map(parse_elements)
            images = processed['image']
            instructions = processed['instructions']
            bboxes = processed['bboxes']
        else:
            # Simple format with one annotation per row
            images = []
            instructions = []
            bboxes = []

            # Group by image
            for example in ds:
                images.append(example['image']) # type: ignore
                instructions.append(example['instructions']) # type: ignore
                bboxes.append(example['bboxes']) # type: ignore


        # Extend combined lists
        all_images.extend(images)
        all_instructions.extend(instructions)
        all_bboxes.extend(bboxes)

    # Create combined dataset
    combined_dataset = ImageAnnotationDataset(
        dataset_name="+".join(dataset_names),
        images=all_images,
        instructions=all_instructions,
        bboxes=all_bboxes,
        bbox_type=DATASETS[dataset_names[0]].bbox_type  # Assume all have same type
    )

    # Split into train/test
    random.seed(seed)
    test_indices = set(random.sample(range(len(combined_dataset)), min(test_size, len(combined_dataset))))

    train_indices = [i for i in range(len(combined_dataset)) if i not in test_indices]
    test_indices = list(test_indices)

    train_dataset = torch.utils.data.Subset(combined_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(combined_dataset, test_indices)

    return train_dataset, test_dataset # type: ignore


def create_message(
    processed_img: ProcessedImage,
    instruction: str,
    bbox: tuple[float, float, float, float],
    format: Literal["plain", "xml", "json"] = "xml",
    include_answer: bool = True
) -> list[dict]:
    """Create a conversation from processed data."""
    x1, y1, x2, y2 = bbox
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

    # Format prompt based on type
    format_prompt = "Return your click as (x, y) pixel coordinates."
    if format == "json":
        format_prompt = "Report the click location as an (x, y) point in JSON format."
    elif format == "xml":
        format_prompt = 'Report the click location in XML like <points x1="x" y1="y">object</points>.'

    messages = [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": (
                    "Determine where to click in the UI to complete the task. "
                    + format_prompt
                    + f" The image is size [{processed_img.width}, {processed_img.height}], "
                    + f"(0, 0) is the top-left and ({processed_img.width}, {processed_img.height}) is the bottom-right."
                ),
            }],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": processed_img.data_uri},
                {"type": "text", "text": instruction},
            ],
        }
    ]

    if include_answer:
        label = format_point((cx, cy), format=format)
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": label}]
        })

    return messages


def collate_fn(
    batch: list[dict[str, Any]],
    processor: Any,
    eval: bool = False,
    message_format: Literal["plain", "xml", "json"] = "xml"
) -> dict[str, torch.Tensor]:
    """Collate function that creates messages on-the-fly."""
    conversations = []
    bboxes = []

    for item in batch:
        # Create message for this item
        messages = create_message(
            item['processed_image'],
            item['instruction'],
            item['bbox'],
            format=message_format,
            include_answer=not eval
        )
        conversations.append(messages)
        bboxes.append(item['bbox'])

    # Process with Qwen processor
    texts = [
        processor.apply_chat_template(conv, add_generation_prompt=eval, tokenize=False)
        for conv in conversations
    ]

    image_inputs, _ = process_vision_info(conversations) # type: ignore

    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=None,
        padding=True,
        return_tensors="pt",
    )

    # Add labels for training
    if not eval:
        inputs = add_labels_to_inputs(inputs)
    else:
        inputs["bbox"] = bboxes  # Include ground truth for evaluation

    return inputs


def add_labels_to_inputs(
    inputs: dict[str, torch.Tensor],
    IM_START: int = 151644,
    ASSISTANT: int = 77091
) -> dict[str, torch.Tensor]:
    """Add labels, masking everything except assistant response."""
    labels = inputs["input_ids"].clone()

    for i, ids in enumerate(inputs["input_ids"]):
        # Find the last "<|im_start|> assistant" pair
        starts = (ids == IM_START).nonzero(as_tuple=True)[0]
        tgt_start = None
        for s in reversed(starts.tolist()):
            if s + 1 < len(ids) and ids[s + 1] == ASSISTANT:
                tgt_start = s + 2  # First token after "assistant"
                break

        if tgt_start is None:
            raise ValueError("No assistant response found in input")
        else:
            labels[i, :tgt_start] = -100  # Ignore everything before assistant response

    inputs["labels"] = labels
    return inputs
