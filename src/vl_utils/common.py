import math

import torch

# Constants from Qwen2.5-VL
IMAGE_FACTOR = 28  # Core patch size factor
MIN_PIXELS = 4 * 28 * 28  # Minimum pixels (3,136)
MAX_PIXELS = 1_024 * 1_024  # 16384 * 28 * 28  # Maximum pixels (12,845,056)
MAX_RATIO = 200  # Maximum aspect ratio allowed

# Vision tokens special IDs (you'll need these from the tokenizer)
VISION_START_TOKEN_ID = 151655  # <|vision_start|>
VISION_END_TOKEN_ID = 151656  # <|vision_end|>
IMAGE_PAD_TOKEN_ID = 151657  # <|image_pad|>


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    """
    Smartly resize image dimensions to fit within constraints while maintaining aspect ratio.
    This is the core algorithm Qwen2.5-VL uses for dynamic resolution.
    """
    # Check aspect ratio
    h_bar = max(height, width)
    w_bar = min(height, width)

    if h_bar / w_bar > MAX_RATIO:
        raise ValueError(f"Aspect ratio {h_bar / w_bar} exceeds maximum {MAX_RATIO}")

    # Initial resize to fit within bounds
    if height * width > max_pixels:
        # Scale down
        scale = math.sqrt(max_pixels / (height * width))
        height = int(height * scale)
        width = int(width * scale)

    elif height * width < min_pixels:
        # Scale up
        scale = math.sqrt(min_pixels / (height * width))
        height = int(height * scale)
        width = int(width * scale)

    # Round to nearest factor
    height = round_by_factor(height, factor)
    width = round_by_factor(width, factor)

    # Ensure we're still within bounds after rounding
    if height * width > max_pixels:
        # If rounding pushed us over, round down instead
        height = floor_by_factor(height, factor)
        width = floor_by_factor(width, factor)

    return height, width


def create_image_tokens(num_patches: int, device="cpu"):
    """
    Create the token sequence for an image.
    This replaces the image placeholder in the text with actual vision tokens.
    """
    # The pattern is: <|vision_start|> <|image_pad|> * num_patches <|vision_end|>
    tokens = [VISION_START_TOKEN_ID]
    tokens.extend([IMAGE_PAD_TOKEN_ID] * num_patches)
    tokens.append(VISION_END_TOKEN_ID)

    return torch.tensor(tokens, device=device)
