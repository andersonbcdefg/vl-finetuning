import base64
from io import BytesIO
from typing import cast

import requests
import torch
from PIL import Image
from torchvision import transforms as T
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .common import MAX_PIXELS, MIN_PIXELS, smart_resize


def load_image(image_input: Image.Image | str):
    """Load image from various sources (path, URL, base64, PIL.Image)."""
    if isinstance(image_input, Image.Image):
        return image_input

    elif isinstance(image_input, str):
        if image_input.startswith("http://") or image_input.startswith("https://"):
            # URL
            response = requests.get(image_input, stream=True)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))

        elif image_input.startswith("data:image"):
            # Base64
            if "base64," in image_input:
                _, base64_data = image_input.split("base64,", 1)
                data = base64.b64decode(base64_data)
                return Image.open(BytesIO(data))

        else:
            # File path
            return Image.open(image_input)

    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")


def preprocess_image(
    image_input: Image.Image | str,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
    target_height: int | None = None,
    target_width: int | None = None,
    return_tensors: str = "pt",
):
    """
    Complete preprocessing pipeline for a single image.
    This replaces what AutoProcessor does internally.
    """
    # 1. Load and convert to RGB
    image = load_image(image_input)
    assert image, "no image loaded"
    if image.mode != "RGB":
        image = image.convert("RGB")

    # 2. Get current dimensions
    width, height = image.size

    # 3. Calculate target dimensions
    if target_height is not None and target_width is not None:
        # Use specified dimensions
        new_height, new_width = target_height, target_width
    else:
        # Use smart resize algorithm
        new_height, new_width = smart_resize(
            height, width, min_pixels=min_pixels, max_pixels=max_pixels
        )

    # 4. Resize image using bicubic interpolation (same as Qwen)
    if (new_height, new_width) != (height, width):
        image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)

    # 5. Convert to tensor and normalize
    # Qwen uses standard ImageNet normalization
    _PRE_TF = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],  # CLIP normalization
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )

    pixel_values = cast(torch.Tensor, _PRE_TF(image))

    # 6. Calculate number of image patches (for token generation)
    # Qwen2.5-VL uses 14x14 patches for the ViT
    patches_height = new_height // 14
    patches_width = new_width // 14
    num_patches = patches_height * patches_width

    if return_tensors == "pt":
        pixel_values = pixel_values.unsqueeze(0)  # Add batch dimension

    return {
        "pixel_values": pixel_values,
        "image_size": (new_height, new_width),
        "num_patches": num_patches,
        "patches_grid": (patches_height, patches_width),
    }


def _tokenize_text(text, tokenizer: PreTrainedTokenizerBase):
    encoding = tokenizer.apply_chat_template(text)


def process_messages_with_images(
    messages, tokenizer, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
):
    """
    Process a list of messages containing text and images.
    This replaces process_vision_info + processor workflow.
    """
    all_pixel_values = []
    all_image_sizes = []
    all_num_patches = []

    # Process each message
    for message in messages:
        if message["role"] == "user" and isinstance(message["content"], list):
            for content in message["content"]:
                if content["type"] == "image":
                    # Process image
                    result = preprocess_image(
                        content["image"], min_pixels=min_pixels, max_pixels=max_pixels
                    )
                    all_pixel_values.append(result["pixel_values"])
                    all_image_sizes.append(result["image_size"])
                    all_num_patches.append(result["num_patches"])
                elif content["type"] == "text":
                    pass
                else:
                    raise ValueError(f"unsupported content type {content['type']}")

    # Stack all images
    if all_pixel_values:
        pixel_values = torch.cat(all_pixel_values, dim=0)
    else:
        pixel_values = None

    return {
        "pixel_values": pixel_values,
        "image_sizes": all_image_sizes,
        "num_patches": all_num_patches,
    }


# Example usage for your specific case:
def preprocess_webclick_batch(batch, tokenizer):
    """
    Process a batch from the WebClick dataset without using AutoProcessor.
    """
    processed_images = []
    processed_texts = []

    for img, inst, bbox in zip(batch["image"], batch["instruction"], batch["bbox"]):
        # Calculate click position (center of bbox)
        x1, y1, x2, y2 = bbox
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        label = f"({x:.3f}, {y:.3f})"

        # Preprocess image
        img_result = preprocess_image(img.convert("RGB"))
        processed_images.append(img_result["pixel_values"])

        # Create message format
        messages = [
            {
                "role": "system",
                "content": "Determine where to click in the UI to complete the task. "
                "Return your click as (x, y) with each coordinate in [0, 1], "
                "where (0,0) is the top-left, to 3 decimal places.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "placeholder",
                    },  # Will be replaced with tokens
                    {"type": "text", "text": inst},
                ],
            },
            {"role": "assistant", "content": label},
        ]

        # Apply chat template (you'll need to implement this based on Qwen's format)
        # This is a simplified version - you'd need the actual template
        text = f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n"
        text += f"<|im_start|>user\n<|vision_start|><|image_pad|>*{img_result['num_patches']}<|vision_end|>{inst}<|im_end|>\n"
        text += f"<|im_start|>assistant\n{label}<|im_end|>"

        processed_texts.append(text)

    # Stack images
    pixel_values = torch.cat(processed_images, dim=0)

    # Tokenize texts (simplified - you'd use the actual tokenizer)
    # tokens = tokenizer(processed_texts, padding=True, return_tensors="pt")

    return {
        "pixel_values": pixel_values,
        "texts": processed_texts,
        # "input_ids": tokens["input_ids"],
        # "attention_mask": tokens["attention_mask"]
    }


# The actual forward pass would look something like:
def forward_pass_example(model, pixel_values, input_ids, attention_mask):
    """
    Example of how the preprocessed data flows through the model.
    """
    # 1. Vision encoder processes the images
    # This happens inside the model, but conceptually:
    # vision_features = model.vision_model(pixel_values)

    # 2. Vision adapter projects features to language model dimension
    # adapted_features = model.vision_adapter(vision_features)

    # 3. Replace image pad tokens in input_ids with vision features
    # The model internally handles merging vision and text features

    # 4. Run through language model
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        # image_sizes=image_sizes  # Some versions use this
    )

    return outputs


if __name__ == "__main__":
    # Test the preprocessing
    test_image = Image.new("RGB", (1920, 1080), color="red")
    result = preprocess_image(test_image)

    print("Original size: 1920x1080")
    print(f"Processed size: {result['image_size']}")
    print(f"Number of patches: {result['num_patches']}")
    print(f"Patches grid: {result['patches_grid']}")
    print(f"Pixel values shape: {result['pixel_values'].shape}")
