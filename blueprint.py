import datetime
import io

import cv2
import numpy as np
from PIL import Image

BLUE_BG = (110, 46, 10)
BLUE_GRID = (95, 40, 8)
BLUE_GRID_FINE = (85, 36, 7)
WHITE = (255, 255, 255)
LIGHT_BLUE = (255, 200, 120)
CYAN = (255, 220, 0)
DIM = (200, 200, 200)


def format_len(mm, unit):
    if unit == "cm":
        return f"{mm / 10:.1f} cm"
    if unit == "m":
        return f"{mm / 1000:.2f} m"
    return f"{mm:.0f} mm"


def _find_object_rects(twin, exclude_mask=None):
    gray = cv2.cvtColor(twin, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = gray.shape
    sheet_area = h * w
    sobx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    soby = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(sobx ** 2 + soby ** 2)

    def passes(c):
        area = cv2.contourArea(c)
        if not (0.003 * sheet_area < area < 0.95 * sheet_area):
            return None
        x, y, rw, rh = cv2.boundingRect(c)
        if rw < 10 or rh < 10:
            return None
        if area < 0.3 * rw * rh:
            return None
        if rw > 0.9 * w and rh > 0.9 * h:
            return None
        pts = c.reshape(-1, 2)
        if float(grad[pts[:, 1], pts[:, 0]].mean()) < 55:
            return None
        return (x, y, rw, rh)

    def inter(a, b):
        ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
        iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
        return ix * iy

    rects = []
    for mask in (255 - bw, bw):
        if exclude_mask is not None:
            mask = mask & ~exclude_mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            r = passes(c)
            if r is not None:
                rects.append(r)
    rects.sort(key=lambda r: r[2] * r[3], reverse=True)
    kept = []
    for r in rects:
        if any(abs(r[0] - k[0]) < 8 and abs(r[1] - k[1]) < 8
               and abs(r[2] - k[2]) < 8 and abs(r[3] - k[3]) < 8 for k in kept):
            continue
        kept.append(r)

    edges = cv2.Canny(gray, 25, 75)
    if exclude_mask is not None:
        edges = edges & ~exclude_mask
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                              np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    extra = []
    for c in contours:
        r = passes(c)
        if r is None:
            continue
        x, y, rw, rh = r
        if rw <= 14 or rh <= 14:
            continue
        r = (x + 7, y + 7, rw - 14, rh - 14)
        if any(inter(r, k) > 0.5 * k[2] * k[3] for k in kept):
            continue
        if any(inter(r, k) > 0.5 * min(r[2] * r[3], k[2] * k[3])
               for k in extra):
            continue
        extra.append(r)
    kept.extend(extra)
    kept.sort(key=lambda r: r[2] * r[3], reverse=True)
    return kept


def _tighten_rect(mask, rect):
    x, y, w, h = rect
    mh, mw = mask.shape[:2]

    def side_fill_top(r):
        band = mask[max(0, r):min(mh, r + 3), x:x + w]
        return float(band.mean()) if band.size else 0.0

    def side_fill_bottom(r):
        band = mask[max(0, r - 2):min(mh, r + 1), x:x + w]
        return float(band.mean()) if band.size else 0.0

    def side_fill_left(c):
        band = mask[y:y + h, max(0, c):min(mw, c + 3)]
        return float(band.mean()) if band.size else 0.0

    def side_fill_right(c):
        band = mask[y:y + h, max(0, c - 2):min(mw, c + 1)]
        return float(band.mean()) if band.size else 0.0

    max_trim_x, max_trim_y = int(0.25 * w), int(0.25 * h)
    top = y
    while top < y + max_trim_y and side_fill_top(top) < 0.5:
        top += 1
    bottom = y + h - 1
    while bottom > y + h - 1 - max_trim_y and side_fill_bottom(bottom) < 0.5:
        bottom -= 1
    left = x
    while left < x + max_trim_x and side_fill_left(left) < 0.5:
        left += 1
    right = x + w - 1
    while right > x + w - 1 - max_trim_x and side_fill_right(right) < 0.5:
        right -= 1
    if right <= left or bottom <= top:
        return rect
    return (left, top, right - left + 1, bottom - top + 1)


def _mask_is_dark(twin, x, y, w, h):
    gray = cv2.cvtColor(twin, cv2.COLOR_BGR2GRAY)
    return float(gray[y:y + h, x:x + w].mean()) < 128


def _find_object_rect(twin):
    rects = _find_object_rects(twin)
    return rects[0] if rects else None


def _inner_rects(twin):
    gray = cv2.cvtColor(twin, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    out = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 200 or area > 0.25 * h * w:
            continue
        x, y, rw, rh = cv2.boundingRect(c)
        if rw < 8 or rh < 8:
            continue
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) < 4:
            continue
        out.append((x, y, rw, rh))
    out.sort(key=lambda r: r[2] * r[3], reverse=True)
    kept = []
    for r in out:
        if any(abs(r[0] - k[0]) < 6 and abs(r[1] - k[1]) < 6
               and abs(r[2] - k[2]) < 6 and abs(r[3] - k[3]) < 6 for k in kept):
            continue
        kept.append(r)
        if len(kept) >= 60:
            break
    return kept


def _dashed_line(img, p1, p2, color, thickness=1, dash=14, gap=10):
    x1, y1, x2, y2 = *p1, *p2
    length = int(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
    if length == 0:
        return
    dx, dy = (x2 - x1) / length, (y2 - y1) / length
    d = 0
    while d < length:
        e = min(d + dash, length)
        cv2.line(img, (int(x1 + dx * d), int(y1 + dy * d)),
                 (int(x1 + dx * e), int(y1 + dy * e)), color, thickness)
        d = e + gap


def _dim_h(img, x0, x1, y, label, color=CYAN):
    cv2.line(img, (x0, y), (x1, y), color, 2, cv2.LINE_AA)
    cv2.arrowedLine(img, (x0, y), (x0 + 14, y), color, 2, tipLength=0.6,
                    line_type=cv2.LINE_AA)
    cv2.arrowedLine(img, (x1, y), (x1 - 14, y), color, 2, tipLength=0.6,
                    line_type=cv2.LINE_AA)
    cv2.line(img, (x0, y - 6), (x0, y + 6), color, 2, cv2.LINE_AA)
    cv2.line(img, (x1, y - 6), (x1, y + 6), color, 2, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    tx = (x0 + x1 - tw) // 2
    cv2.rectangle(img, (tx - 6, y - th - 10), (tx + tw + 6, y + 6),
                  BLUE_BG, -1)
    cv2.putText(img, label, (tx, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, 2, cv2.LINE_AA)


def _dim_v(img, y0, y1, x, label, color=CYAN):
    cv2.line(img, (x, y0), (x, y1), color, 2, cv2.LINE_AA)
    cv2.arrowedLine(img, (x, y0), (x, y0 + 14), color, 2, tipLength=0.6,
                    line_type=cv2.LINE_AA)
    cv2.arrowedLine(img, (x, y1), (x, y1 - 14), color, 2, tipLength=0.6,
                    line_type=cv2.LINE_AA)
    cv2.line(img, (x - 6, y0), (x + 6, y0), color, 2, cv2.LINE_AA)
    cv2.line(img, (x - 6, y1), (x + 6, y1), color, 2, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    txt = np.zeros((th + 16, tw + 8, 3), dtype=np.uint8)
    txt[:] = BLUE_BG
    cv2.putText(txt, label, (4, th + 6), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, 2, cv2.LINE_AA)
    txt = cv2.rotate(txt, cv2.ROTATE_90_COUNTERCLOCKWISE)
    th2, tw2 = txt.shape[:2]
    y = (y0 + y1 - th2) // 2
    img[max(0, y):y + th2, x - tw2 // 2:x - tw2 // 2 + tw2] = txt


def blueprint_pdf_bytes(sheet, px_per_mm):
    rgb = cv2.cvtColor(sheet, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, "PDF", resolution=float(px_per_mm) * 25.4)
    return buf.getvalue()


def save_blueprint_pdf(sheet, px_per_mm, path):
    with open(path, "wb") as f:
        f.write(blueprint_pdf_bytes(sheet, px_per_mm))


def make_blueprint(twin, width_mm, height_mm, px_per_mm,
                   title="CALCULATOR", exclude=None, unit="mm"):
    if unit not in ("mm", "cm", "m"):
        unit = "mm"
    """twin (BGR) -> (blueprint sheet (BGR), info dict).

    The sheet is the twin plus margins: 90 px left/top for the
    dimension lines, 90 px bottom for the title block.
    """
    th, tw = twin.shape[:2]
    mx, my, bottom = 90, 90, 90
    sheet = np.zeros((th + my + bottom, tw + mx + 70, 3), dtype=np.uint8)
    sheet[:] = BLUE_BG
    ox, oy = mx, my

    step = max(1, int(round(10 * px_per_mm)))
    for x in range(0, sheet.shape[1], step):
        cv2.line(sheet, (x, 0), (x, sheet.shape[0]), BLUE_GRID_FINE, 1)
    for y in range(0, sheet.shape[0], step):
        cv2.line(sheet, (0, y), (sheet.shape[1], y), BLUE_GRID_FINE, 1)
    step5 = max(1, int(round(50 * px_per_mm)))
    for x in range(0, sheet.shape[1], step5):
        cv2.line(sheet, (x, 0), (x, sheet.shape[0]), BLUE_GRID, 1)
    for y in range(0, sheet.shape[0], step5):
        cv2.line(sheet, (0, y), (sheet.shape[1], y), BLUE_GRID, 1)

    gray = cv2.cvtColor(twin, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 25, 75)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (edges > 0).astype(np.uint8), 8)
    keep = np.zeros(edges.shape, bool)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 40:
            keep[labels == i] = True
    edges = np.where(keep, 255, 0).astype(np.uint8)
    sheet[oy:oy + th, ox:ox + tw][edges > 0] = WHITE

    cv2.rectangle(sheet, (ox, oy), (ox + tw, oy + th), WHITE, 2)

    gray_obj = cv2.cvtColor(twin, cv2.COLOR_BGR2GRAY)
    gray_obj = cv2.GaussianBlur(gray_obj, (5, 5), 0)
    _, bw_obj = cv2.threshold(gray_obj, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    exclude_mask = None
    if exclude:
        exclude_mask = np.zeros((th, tw), dtype=np.uint8)
        for (p1, p2, half_w) in exclude:
            cv2.line(exclude_mask,
                     (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])),
                     255, max(1, int(2 * half_w)))
    objects = _find_object_rects(twin, exclude_mask)
    if not objects:
        objects = [(0, 0, tw, th)]
    rects = []
    obj_infos = []
    for (x, y, w, h) in objects:
        mask = np.zeros((th, tw), dtype=np.uint8)
        contours, _ = cv2.findContours(
            (255 - bw_obj) if _mask_is_dark(twin, x, y, w, h) else bw_obj,
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for c in contours:
            cx, cy, cw, ch = cv2.boundingRect(c)
            if abs(cx - x) < 8 and abs(cy - y) < 8 and abs(cw - w) < 8 and abs(ch - h) < 8:
                best = c
                break
        if best is not None:
            solid = cv2.contourArea(best) / max(1.0, w * h)
            cv2.drawContours(mask, [best], -1, 1, cv2.FILLED)
            if solid > 0.8:
                x, y, w, h = _tighten_rect(mask, (x, y, w, h))
        rects.append((x, y, w, h))
        obj_w_mm = w / px_per_mm
        obj_h_mm = h / px_per_mm
        obj_infos.append({"w_mm": round(obj_w_mm, 1),
                          "h_mm": round(obj_h_mm, 1),
                          "px": [int(x), int(y), int(w), int(h)]})

    placed_w, placed_h = [], []

    def _place_w(base_y, x0, x1, direction):
        lvl = 0
        while True:
            y = base_y + direction * lvl * 40
            if direction < 0 and y < 15:
                return None
            clash = any(abs(y - py) < 10 and x0 - 15 < px1 and px0 < x1 + 15
                        for (py, px0, px1) in placed_w)
            if not clash:
                placed_w.append((y, x0, x1))
                return y
            lvl += 1

    def _place_h(base_x, y0, y1, direction):
        lvl = 0
        while True:
            x = base_x + direction * lvl * 40
            if direction < 0 and x < 15:
                return None
            if direction > 0 and x > sheet.shape[1] - 15:
                return None
            clash = any(abs(x - px) < 10 and y0 - 15 < py1 and py0 < y1 + 15
                        for (px, py0, py1) in placed_h)
            if not clash:
                placed_h.append((x, y0, y1))
                return x
            lvl += 1
    for (x, y, w, h), info in zip(rects, obj_infos):
        cv2.rectangle(sheet, (ox + x, oy + y), (ox + x + w, oy + y + h),
                      WHITE, 3, cv2.LINE_AA)
        cx, cy = ox + x + w // 2, oy + y + h // 2
        _dashed_line(sheet, (ox + x - 30, cy), (ox + x + w + 30, cy), DIM)
        _dashed_line(sheet, (cx, oy + y - 30), (cx, oy + y + h + 30), DIM)
        dim_y = _place_w(oy + y - 30, ox + x, ox + x + w, -1)
        below = dim_y is None
        if below:
            dim_y = _place_w(oy + y + h + 30, ox + x, ox + x + w, +1)
        if below:
            cv2.line(sheet, (ox + x, oy + y + h), (ox + x, dim_y + 5), CYAN, 1)
            cv2.line(sheet, (ox + x + w, oy + y + h), (ox + x + w, dim_y + 5), CYAN, 1)
        else:
            cv2.line(sheet, (ox + x, oy + y), (ox + x, dim_y - 5), CYAN, 1)
            cv2.line(sheet, (ox + x + w, oy + y), (ox + x + w, dim_y - 5), CYAN, 1)
        _dim_h(sheet, ox + x, ox + x + w, dim_y, format_len(info["w_mm"], unit))
        side = "left" if cx < ox + tw // 2 else "right"
        if side == "left":
            dim_x = _place_h(ox + x - 30, oy + y, oy + y + h, -1)
            if dim_x is None:
                side = "right"
                dim_x = _place_h(ox + x + w + 30, oy + y, oy + y + h, +1)
        else:
            dim_x = _place_h(ox + x + w + 30, oy + y, oy + y + h, +1)
            if dim_x is None:
                side = "left"
                dim_x = _place_h(ox + x - 30, oy + y, oy + y + h, -1)
        if side == "left":
            cv2.line(sheet, (ox + x, oy + y), (dim_x + 5, oy + y), CYAN, 1)
            cv2.line(sheet, (ox + x, oy + y + h), (dim_x + 5, oy + y + h), CYAN, 1)
        else:
            cv2.line(sheet, (ox + x + w, oy + y), (dim_x - 5, oy + y), CYAN, 1)
            cv2.line(sheet, (ox + x + w, oy + y + h), (dim_x - 5, oy + y + h), CYAN, 1)
        _dim_v(sheet, oy + y, oy + y + h, dim_x, format_len(info["h_mm"], unit))

    for (ix, iy, iw, ih) in _inner_rects(twin):
        cv2.rectangle(sheet, (ox + ix, oy + iy), (ox + ix + iw, oy + iy + ih),
                      LIGHT_BLUE, 1)

    cv2.rectangle(sheet, (8, 8), (sheet.shape[1] - 9, sheet.shape[0] - 9),
                  WHITE, 2)
    ty = sheet.shape[0] - bottom + 12
    cv2.line(sheet, (8, ty - 8), (sheet.shape[1] - 9, ty - 8), WHITE, 2)
    title_text = f"{title}  —  TOP VIEW"
    if len(obj_infos) > 1:
        title_text += f"  ·  {len(obj_infos)} OBJECTS"
    cv2.putText(sheet, title_text, (24, ty + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2, cv2.LINE_AA)
    today = datetime.date.today().isoformat()
    if unit == "cm":
        ref_txt = f"{width_mm / 10:.1f} x {height_mm / 10:.1f} cm"
    elif unit == "m":
        ref_txt = f"{width_mm / 1000:.2f} x {height_mm / 1000:.2f} m"
    else:
        ref_txt = f"{width_mm:.0f} x {height_mm:.0f} mm"
    cv2.putText(sheet,
                f"REF {ref_txt}  |  "
                f"SCALE {px_per_mm:g} px/mm  |  "
                f"GRID {format_len(10, unit)}  |  {today}",
                (24, ty + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, CYAN, 1,
                cv2.LINE_AA)
    cv2.putText(sheet, "BLUEPRINT", (sheet.shape[1] - 190, ty + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, CYAN, 2, cv2.LINE_AA)

    biggest = obj_infos[0]
    info = {
        "object_w_mm": biggest["w_mm"],
        "object_h_mm": biggest["h_mm"],
        "object_px": biggest["px"],
        "objects": obj_infos,
        "object_count": len(obj_infos),
        "unit": unit,
        "sheet_px": [int(sheet.shape[1]), int(sheet.shape[0])],
        "ref_w_mm": width_mm,
        "ref_h_mm": height_mm,
        "px_per_mm": px_per_mm,
    }
    return sheet, info
