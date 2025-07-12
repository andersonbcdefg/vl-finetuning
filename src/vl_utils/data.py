import base64
import io

from PIL import Image, ImageFile
from qwen_vl_utils import smart_resize
from typing import Literal

from .common import MAX_PIXELS
from .spatial import format_point


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

        label = format_point(cx, cy)
        format_prompt = "Return your click as (x, y) pixel coordinates."
        if format == "json":
            format_prompt = "Report the click location as an (x, y) point in JSON format."
        elif format == "xml":
            format_prompt = "Report the click location in XML like <points x y>object</points>."

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
