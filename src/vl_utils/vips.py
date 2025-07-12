import base64
from typing import List, Union, cast

import numpy as np
import pyvips
import requests
import torch

from .common import smart_resize

# Vision tokens special IDs (you'll need these from the tokenizer)
VISION_START_TOKEN_ID = 151655  # <|vision_start|>
VISION_END_TOKEN_ID = 151656  # <|vision_end|>
IMAGE_PAD_TOKEN_ID = 151657  # <|image_pad|>


def _ensure_single(img: Union[pyvips.Image, List[pyvips.Image], None]) -> pyvips.Image:
    """Return a single pyvips.Image, or raise if that’s impossible."""
    if img is None:
        raise ValueError("Could not load image bytes into pyvips.")
    if isinstance(img, list):
        if not img:
            raise ValueError("Loader returned an empty image list.")
        img = img[0]  # pick the first page/frame
    return cast(pyvips.Image, img)  # convince the type checker


def load_image_vips(image_input: Union[str, pyvips.Image]) -> pyvips.Image:
    """
    Load an image (file path, URL, data-URI, or pyvips.Image) and return a pyvips.Image.
    Always returns **one** image: for multi-page inputs we take the first page.
    """
    if isinstance(image_input, pyvips.Image):
        return image_input

    if isinstance(image_input, str):
        # Remote URL
        if image_input.startswith(("http://", "https://")):
            resp = requests.get(image_input, stream=True)
            resp.raise_for_status()
            return _ensure_single(
                pyvips.Image.new_from_buffer(resp.content, "", access="sequential")
            )

        # data URI
        if image_input.startswith("data:image"):
            _, b64 = image_input.split("base64,", 1)
            data = base64.b64decode(b64)
            return _ensure_single(
                pyvips.Image.new_from_buffer(data, "", access="sequential")
            )

        # Local file path
        return _ensure_single(
            pyvips.Image.new_from_file(image_input, access="sequential")
        )

    raise TypeError(f"Unsupported image input type: {type(image_input)}")


# Image-net / CLIP normalisation
_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
_CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)


def preprocess_image_vips(
    image_input: Union[str, pyvips.Image],
    min_pixels: int,
    max_pixels: int,
    target_height: int | None = None,
    target_width: int | None = None,
    return_tensors: str = "pt",
):
    # ── Load & guarantee correct static type ─────────────────────────────────
    if isinstance(image_input, str):
        image = load_image_vips(image_input)  # ← already single Image
    else:
        image = image_input

    # sRGB, drop alpha
    assert image, "no image loaded"
    if image.interpretation != pyvips.enums.Interpretation.SRGB:
        image = image.colourspace("srgb")  # type: ignore
    if image.bands > 3:  # type: ignore
        image = image.extract_band(0, n=3)  # type: ignore[arg-type]
    assert isinstance(image, pyvips.Image)
    w, h = cast(int, image.width), cast(int, image.height)

    # ── target size ──────────────────────────────────────────────────────────
    if target_height is None or target_width is None:
        target_height, target_width = smart_resize(h, w, min_pixels, max_pixels)

    if (target_height, target_width) != (h, w):
        sx, sy = target_width / w, target_height / h
        image = image.resize(sx, vscale=sy, kernel="bicubic")  # type: ignore[arg-name]

    # ── to Torch tensor, normalise ───────────────────────────────────────────
    assert isinstance(image, pyvips.Image)
    buf = image.write_to_memory()
    np_img = np.frombuffer(buf, dtype=np.uint8).reshape(
        image.height,  # type: ignore
        image.width,  # type: ignore
        image.bands,  # type: ignore
    )
    tensor = torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0
    pixel_values = (tensor - _CLIP_MEAN) / _CLIP_STD
    if return_tensors == "pt":
        pixel_values = pixel_values.unsqueeze(0)

    # ── patch grid (14×14) ───────────────────────────────────────────────────
    ph, pw = target_height // 14, target_width // 14
    return {
        "pixel_values": pixel_values,
        "image_size": (target_height, target_width),
        "num_patches": ph * pw,
        "patches_grid": (ph, pw),
    }
