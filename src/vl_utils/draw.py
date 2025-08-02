import io
from PIL import Image, ImageDraw
from typing import TypeAlias, Union
import math

# BBox = tuple[float, float, float, float]  # (x, y, w, h) in normalised [0, 1] coords
# define bbox type
BBox: TypeAlias = Union[tuple[float, float, float, float], tuple[int, int, int, int]]

def rgba(hex_rgb: str, alpha: int = 80) -> tuple[int, int, int, int]:
    hex_rgb = hex_rgb.lstrip("#")
    r, g, b = (int(hex_rgb[i:i + 2], 16) for i in (0, 2, 4))
    return r, g, b, alpha

def draw_arrow(draw: ImageDraw.ImageDraw,
                tail: tuple[int, int],
                head: tuple[int, int],
                color: str,
                width: int = 4,
                head_len: int = 16,
                head_angle: float = 28) -> None:
    """Draw a line plus filled arrowhead (always pointing tail → head)."""
    draw.line([tail, head], fill=color, width=width)

    theta = math.atan2(tail[1] - head[1], tail[0] - head[0])
    left = (head[0] + head_len * math.cos(theta + math.radians(head_angle)),
            head[1] + head_len * math.sin(theta + math.radians(head_angle)))
    right = (head[0] + head_len * math.cos(theta - math.radians(head_angle)),
             head[1] + head_len * math.sin(theta - math.radians(head_angle)))
    draw.polygon([head, left, right], fill=color)

def edge_intersection(cx: float, cy: float,
                       dx: float, dy: float,
                       x1: int, y1: int, x2: int, y2: int) -> tuple[int, int]:
    """
    First hit of the ray (cx,cy) + t·(dx,dy) (t ≥ 0) with rectangle border.
    (dx,dy) should be a normalised direction pointing OUT of the box.
    """
    hits = []
    if dx:  # vertical edges
        t = (x1 - cx) / dx
        y = cy + dy * t
        if t >= 0 and y1 <= y <= y2:
            hits.append(t)
        t = (x2 - cx) / dx
        y = cy + dy * t
        if t >= 0 and y1 <= y <= y2:
            hits.append(t)
    if dy:  # horizontal edges
        t = (y1 - cy) / dy
        x = cx + dx * t
        if t >= 0 and x1 <= x <= x2:
            hits.append(t)
        t = (y2 - cy) / dy
        x = cx + dx * t
        if t >= 0 and x1 <= x <= x2:
            hits.append(t)
    t_hit = min(hits) if hits else 0
    return int(cx + dx * t_hit), int(cy + dy * t_hit)

def minmax_to_pixels(b: BBox, W: int, H: int) -> tuple[int, int, int, int]:
    """Return absolute pixel (x1, y1, x2, y2) from either format."""
    x1, y1, x2, y2 = b
    if max(x1, y1, x2, y2) <= 1.5:         # heuristically: normalised
        return (
            int(x1 * W),
            int(y1 * H),
            int(x2 * W),
            int(y2 * H),
        )
    # already in pixels
    return int(x1), int(y1), int(x2), int(y2)

def draw_overlay(
    im: Image.Image,
    bbox: BBox,
    *,
    stroke: int = 3,
    outline_rgba: tuple[int, int, int, int] = (255, 0, 0, 192),
    fill_rgba: tuple[int, int, int, int] = (255, 0, 0, 32),
) -> None:
    """
    Draw translucent rectangle plus context‑aware arrow, **in place** on *im*.
    """
    # Convert to RGBA if needed for proper alpha blending
    if im.mode != "RGBA":
        im_rgba = im.convert("RGBA")
    else:
        im_rgba = im

    # Create transparent overlay for blending
    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, mode="RGBA")

    W, H = im.size
    x1, y1, x2, y2 = minmax_to_pixels(bbox, W, H)

    # Add internal padding to avoid occluding content
    internal_padding = 3
    padded_x1, padded_y1 = x1 - internal_padding, y1 - internal_padding
    padded_x2, padded_y2 = x2 + internal_padding, y2 + internal_padding

    # Rectangle (using padded coordinates)
    draw.rectangle([padded_x1, padded_y1, padded_x2, padded_y2], fill=fill_rgba,
                   outline=outline_rgba, width=stroke)

    # Box centre
    bx, by = (x1 + x2) / 2, (y1 + y2) / 2
    cx, cy = W / 2, H / 2                    # image centre

    arrow_len = max(int(0.15 * min(W, H)), 250)        # much longer arrows so shaft is visible
    # print("arrowlen:", arrow_len)
    # Vector from image centre → box centre
    dx, dy = bx - cx, by - cy
    dist = math.hypot(dx, dy)
    if dist <= 1e-3:                         # box near centre
        u = (0.0, -1.0)                      # point upward
    else:
        u = (dx / dist, dy / dist)

    # Decide where arrow tail goes
    edge_proximity = max(abs(dx) / (W / 2), abs(dy) / (H / 2))
    if edge_proximity >= 0.30:
        # box near an edge → pull arrow tail inward (toward centre)
        tail = (bx - u[0] * arrow_len, by - u[1] * arrow_len)
    else:
        # box near centre → push arrow tail outward toward edge
        tail = (bx + u[0] * arrow_len, by + u[1] * arrow_len)
        tail = (min(max(0, tail[0]), W), min(max(0, tail[1]), H))

    # Arrow head: intersection of ray (bx,by) → tail with rectangle edge, with padding
    wdx, wdy = tail[0] - bx, tail[1] - by
    mag = math.hypot(wdx, wdy) or 1.0
    arrow_padding = 10  # pixels away from box edge
    hx, hy = edge_intersection(bx, by, wdx / mag, wdy / mag,
                                padded_x1 - arrow_padding, padded_y1 - arrow_padding,
                                padded_x2 + arrow_padding, padded_y2 + arrow_padding)

    # --- shaft unit vector (tail -> tip) ---
    sx, sy = hx - tail[0], hy - tail[1]
    smag   = math.hypot(sx, sy) or 1.0
    ux, uy = sx / smag, sy / smag          # unit along the shaft
    px, py = -uy, ux                       # left–hand perpendicular

    head_len   = max(18, stroke * 3.5)       # bigger arrow heads
    half_base  = head_len * 0.6            # half‑width of the base

    # Move arrowhead forward 5px beyond line end for better visibility
    tip_offset = 5
    tip   = (hx + ux * tip_offset, hy + uy * tip_offset)
    base  = (tip[0] - ux * head_len, tip[1] - uy * head_len)
    left  = (base[0] + px * half_base, base[1] + py * half_base)
    right = (base[0] - px * half_base, base[1] - py * half_base)

    # Draw shaft to original intersection point
    draw.line([(tail[0], tail[1]), (hx, hy)],
              fill=outline_rgba, width=stroke)

    draw.polygon([tip, left, right], fill=outline_rgba)

    # Blend the overlay with the original image
    im_rgba = Image.alpha_composite(im_rgba, overlay)

    # Convert back to original mode if needed
    if im.mode != "RGBA":
        result = im_rgba.convert(im.mode)
        im.paste(result)
    else:
        im.paste(im_rgba)

def _to_relative(bbox: BBox, w: int, h: int):
    if max(bbox) <= 1.0:
        return bbox
    x1, y1, x2, y2 = bbox
    return (x1 / w, y1 / h, x2 / w, y2 / h)

def annotate_many(
    base_im: Image.Image,
    bboxes: list[BBox],
    *,
    resize_to: int | None = 1280,
    jpeg_quality: int = 80,
) -> list[bytes]:
    """
    Produce one JPEG per bounding‑box overlay, with a single decode of *base_im*.

    Parameters
    ----------
    base_im
        Already‑decoded RGB `PIL.Image`.
    bboxes
        Iterable of bounding boxes to overlay.
    resize_to
        If set, the image is down‑scaled so its longer side equals this
        size *before* drawing—saves CPU on large originals.
    jpeg_quality
        Quality setting for `Image.save`; lower is faster/smaller.

    Returns
    -------
    list[bytes]
        JPEG‑encoded images, ready to write to disk or feed downstream.
    """
    # convert bboxes to relative
    w, h = base_im.size
    bboxes = [_to_relative(x, w, h) for x in bboxes]

    # resize to 1280px
    if resize_to and max(base_im.size) > resize_to:
        base_im = base_im.copy()
        base_im.thumbnail((resize_to, resize_to), Image.Resampling.LANCZOS)

    outputs: list[bytes] = []
    scratch = io.BytesIO()  # reused buffer to avoid reallocations

    for bbox in bboxes:
        im_copy = base_im.copy()  # cheap COW copy of pixel data
        draw_overlay(im_copy, bbox)

        scratch.seek(0)
        scratch.truncate(0)
        im_copy.save(
            scratch,
            format="JPEG",
            quality=jpeg_quality,
            optimize=False,        # disable slow Huffman optimisation
            progressive=False,     # faster, smaller code path
        )
        outputs.append(scratch.getvalue())

    return outputs

def draw_point(
    img: Image.Image,
    xy: tuple[int, int],
    color: tuple[int, int, int] = (255, 0, 0),
    radius: int = 1
) -> Image.Image:
    """
    Draw a single point (optionally a small filled circle) on a PIL image.

    Args:
        img   : A Pillow Image instance (mode “RGB”, “RGBA”, etc.).
        xy    : tuple (x, y) giving the point’s center.
        color : RGB/RGBA tuple for the point color.  Default red.
        radius: Pixel radius.  1 draws a single pixel with ImageDraw.point;
                >1 draws a filled circle via ImageDraw.ellipse.

    Returns:
        The same Image instance with the point rendered in-place.
    """
    draw = ImageDraw.Draw(img)

    if radius <= 1:
        draw.point(xy, fill=color)
    else:
        x, y = xy
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=color,
            outline=color
        )

    return img

def draw_bboxes(
    image: Image.Image,
    bboxes: list[BBox],
    *,
    outline: tuple[int, int, int] | str = (255, 0, 0),
    width: int = 2
) -> Image.Image:
    """
    Draw axis-aligned bounding boxes on a copy of a PIL image and return it.

    Parameters
    ----------
    image
        Source ``PIL.Image``.
    bboxes
        Iterable of ``(x1, y1, x2, y2)`` pixel coordinates.
    outline
        Color for the box outline (RGB tuple or any Pillow‐accepted color string).
    width
        Line width of the rectangle outline.

    Returns
    -------
    PIL.Image
        A new image with the bounding boxes drawn.
    """
    annotated = image.copy()          # keep the original untouched
    draw = ImageDraw.Draw(annotated)

    for x1, y1, x2, y2 in bboxes:
        draw.rectangle([x1, y1, x2, y2], outline=outline, width=width)

    return annotated
