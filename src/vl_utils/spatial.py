import re
import math
from typing import Literal


def point_in_bbox(point: tuple, bbox: tuple) -> bool:
    px, py = point
    x1, y1, x2, y2 = bbox
    if x1 <= px <= x2 and y1 <= py <= y2:
        return True
    return False

def dist_to_center(point: tuple, bbox: tuple) -> float:
    px, py = point
    x1, y1, x2, y2 = bbox
    return math.sqrt(((px - (x1 + x2) / 2) ** 2) + ((py - (y1 + y2) / 2) ** 2))

def format_point(point: tuple, format: Literal["plain", "xml", "json"]):
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


def parse_point(
    text: str, format: Literal["plain", "xml", "json"]
) -> tuple[float, float] | None:
    """Extract the first point in the specified format; return None if not found."""
    try:
        if format == "plain":
            # Extract from format: "(x,y)"
            m = re.search(r"\(\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\)", text)
            if m:
                return float(m.group(1)), float(m.group(2))

        elif format == "xml":
            # Try new format first: <points x1="x" y1="y" ...>
            m = re.search(r'<points[^>]*x1="([0-9]*\.?[0-9]+)"[^>]*y1="([0-9]*\.?[0-9]+)"', text)
            if m:
                return float(m.group(1)), float(m.group(2))

            # Fall back to old format: <points x y>content</points>
            m = re.search(r"<points\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)>", text)
            if m:
                return float(m.group(1)), float(m.group(2))

        elif format == "json":
            # Extract from format: {"point_2d": [x, y], "label": "click"}
            m = re.search(r'"point_2d":\s*\[\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\]', text)
            if m:
                return float(m.group(1)), float(m.group(2))

        raise ValueError(f"Invalid format: {format}")
    except Exception as e:
        return None
