def xywh_to_xyxy(box):
    """Convert (x, y, w, h) → (x1, y1, x2, y2)."""
    x, y, w, h = box
    return [x, y, x + w, y + h]

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
        new_r = dict(r)          # shallow copy
        new_r["bbox"] = conv(r["bbox"])
        fixed.append(new_r)
    return fixed
