import math
import re
from typing import Literal, TypeAlias, Union

BBox: TypeAlias = Union[tuple[float, float, float, float], tuple[int, int, int, int]]
ImgSize: TypeAlias = tuple[int, int]
Point: TypeAlias = Union[tuple[float, float], tuple[int, int]]


def point_in_bbox(point: Point, bbox: BBox) -> bool:
    px, py = point
    x1, y1, x2, y2 = bbox
    if x1 <= px <= x2 and y1 <= py <= y2:
        return True
    return False


def dist_to_center(point: Point, bbox: BBox) -> float:
    px, py = point
    x1, y1, x2, y2 = bbox
    return math.sqrt(((px - (x1 + x2) / 2) ** 2) + ((py - (y1 + y2) / 2) ** 2))


def format_point(point: Point, format: Literal["plain", "xml", "json"]):
    assert len(point) == 2, "point must be length 2"
    x, y = point
    if format == "plain":
        return f"({x},{y})"
    elif format == "xml":
        return f'<points x1="{x}" y1="{y}">click</points>'
    elif format == "json":
        res = "```json\n[\n"
        res += f'\t{{"point_2d": [{x}, {y}], "label": "click"}}'
        res += "\n}\n```"
        return res
    else:
        raise ValueError(f"unknown format: {format}")


def parse_point(text: str, format: Literal["plain", "xml", "json"]) -> Point | None:
    """Extract the first point in the specified format; return None if not found."""
    try:
        if format == "plain":
            # Extract from format: "(x,y)"
            m = re.search(r"\(\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\)", text)
            if m:
                return float(m.group(1)), float(m.group(2))

        elif format == "xml":
            # Try point format: <point x="x" y="y" alt="...">content</point>
            m = re.search(
                r'<point[^>]*x="([0-9]*\.?[0-9]+)"[^>]*y="([0-9]*\.?[0-9]+)"', text
            )
            if m:
                return float(m.group(1)), float(m.group(2))

            # Try new format: <points x1="x" y1="y" ...>
            m = re.search(
                r'<points[^>]*x1="([0-9]*\.?[0-9]+)"[^>]*y1="([0-9]*\.?[0-9]+)"', text
            )
            if m:
                return float(m.group(1)), float(m.group(2))

            # Fall back to old format: <points x y>content</points>
            m = re.search(r"<points\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)>", text)
            if m:
                return float(m.group(1)), float(m.group(2))

        elif format == "json":
            # Extract from format: {"point_2d": [x, y], "label": "click"}
            m = re.search(
                r'"point_2d":\s*\[\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\]',
                text,
            )
            if m:
                return float(m.group(1)), float(m.group(2))

        raise ValueError(f"Invalid format: {format}")
    except Exception:
        return None


def xywh_to_xyxy(box: BBox) -> BBox:
    """Convert (x, y, w, h) → (x1, y1, x2, y2)."""
    x, y, w, h = box
    return tuple([float(x) for x in [x, y, x + w, y + h]])  # type: ignore


def order_xyxy(box):
    """Sort coords so that x1 ≤ x2 and y1 ≤ y2."""
    x1, y1, x2, y2 = box
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def fix_bboxes(records, assume="xywh"):
    """
    records : np.ndarray of dicts with key 'bbox'
    assume  : 'xywh' or 'unordered'  – choose the hypothesis to apply
    returns : np.ndarray with corrected bboxes
    """
    converters = {"xywh": xywh_to_xyxy, "unordered": order_xyxy}
    conv = converters[assume]
    fixed = []
    for r in records:
        new_r = dict(r)  # shallow copy
        new_r["bbox"] = conv(r["bbox"])
        fixed.append(new_r)
    return fixed


def flat_polygon_to_bbox(poly_flat: list[float]) -> tuple[float, float, float, float]:
    """
    Convert [x1,y1,x2,y2,...] -> [(x1,y1),(x2,y2),...]
    Ignores an odd trailing value if present (defensive).
    """
    xs = [coord for idx, coord in enumerate(poly_flat) if idx % 2 == 0]
    ys = [coord for idx, coord in enumerate(poly_flat) if idx % 2 == 1]

    return min(xs), min(ys), max(xs), max(ys)


def bbox_center(bbox: tuple[float, float, float, float]):
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    return cx, cy


def convert_bbox_format(
    bbox: tuple[float, float, float, float],
    from_format: Literal["xyxy", "xywh"],
    to_format: Literal["xyxy", "xywh"],
) -> tuple[float, float, float, float]:
    """Convert bbox between xyxy and xywh formats."""
    try:
        if from_format == to_format:
            return bbox

        if from_format == "xywh" and to_format == "xyxy":
            x, y, w, h = bbox
            return (x, y, x + w, y + h)
        elif from_format == "xyxy" and to_format == "xywh":
            x1, y1, x2, y2 = bbox
            return (x1, y1, x2 - x1, y2 - y1)
        else:
            raise ValueError(
                f"Unsupported conversion from {from_format} to {to_format}"
            )
    except Exception as e:
        print("failed to convert bbox:", bbox)
        raise e


def absolute_to_relative(
    obj: BBox | Point, size: ImgSize, bbox_format: Literal["xyxy", "xywh"] = "xyxy"
) -> BBox | Point:
    w, h = size
    if len(obj) == 2:
        x, y = obj
        return (x / w, y / h)
    assert len(obj) == 4, "bbox needs to have 4 items"
    if bbox_format == "xyxy":
        x1, y1, x2, y2 = obj
        return (x1 / w, y1 / h, x2 / w, y2 / h)
    elif bbox_format == "xywh":
        x, y, w, h = obj
        return (x / w, y / h, w / w, h / h)
    else:
        raise ValueError(f"Unsupported bbox format: {bbox_format}")


def relative_to_absolute(
    obj: BBox | Point, size: ImgSize, bbox_format: Literal["xyxy", "xywh"] = "xyxy"
) -> BBox | Point:
    w, h = size
    if len(obj) == 2:
        x, y = obj
        return (x * w, y * h)
    assert len(obj) == 4, "bbox needs to have 4 items"
    bbox = obj
    if bbox_format == "xyxy":
        x1, y1, x2, y2 = bbox
        return (x1 * w, y1 * h, x2 * w, y2 * h)
    elif bbox_format == "xywh":
        x, y, w, h = bbox
        return (x * w, y * h, w * w, h * h)
    else:
        raise ValueError(f"Unsupported bbox format: {bbox_format}")


def scale_bbox(
    bbox: BBox,
    bbox_type: Literal["absolute", "relative"],
    bbox_format: Literal["xyxy", "xywh"],
    orig_size: ImgSize,
    new_size: ImgSize,
) -> BBox:
    # converts to scaled bbox relative to NEW image size
    # (NOT between 0 and 1)
    orig_w, orig_h = orig_size
    new_w, new_h = new_size

    # Convert to xyxy format for processing
    xyxy_bbox = None
    try:
        xyxy_bbox = convert_bbox_format(bbox, bbox_format, "xyxy")
        x1, y1, x2, y2 = xyxy_bbox

        if bbox_type == "relative":
            scaled_bbox = [x1 * new_w, y1 * new_h, x2 * new_w, y2 * new_h]
        else:
            w_scale, h_scale = new_w / orig_w, new_h / orig_h
            scaled_bbox = [x1 * w_scale, y1 * h_scale, x2 * w_scale, y2 * h_scale]
    except Exception as e:
        print("failed to scale bbox:", bbox, "got xyxy bbox:", xyxy_bbox)
        raise e

    # VALIDATE!
    try:
        # assert all(0 <= coord <= 1 for coord in scaled_bbox), "Invalid scaled bbox"
        assert scaled_bbox[0] < scaled_bbox[2], "invalid scaled bbox: x1 > x2"
        assert scaled_bbox[1] < scaled_bbox[3], "invalid scaled bbox: y1 > y2"
    except Exception:
        print("WARNING: invalid scaled bbox:", scaled_bbox)

    # Always return xyxy format
    return tuple(scaled_bbox)  # type: ignore
