#!/usr/bin/env python3
"""Twin -> Blueprint: turn a top-down twin photo into a
blueprint drawing, like a submarine blueprint in a game.

  Input : warped top-down twin (BGR), its real size in mm, px/mm
  Output: blueprint sheet (BGR) — same twin, but drawn as:
    * deep-blue paper, faint 10 mm grid
    * white line drawing of the object (edges + inner parts)
    * dimension lines with arrows, labelled in real mm
    * dashed centre lines through the object
    * title block: name, TOP VIEW, scale, reference size

The dimensions are real, not decoration: they come from the
homography scale (px / px_per_mm = mm). Every object on the
sheet gets its own outline and dimensions, measured from its
own contour in the twin — not just the largest one.

Usage:
    from blueprint import make_blueprint
    sheet, info = make_blueprint(twin, 210, 297, 5.0, title="CALCULATOR")
    # info = {"object_w_mm": ..., "object_h_mm": ..., "objects": [...]}
"""
import datetime
import io

import cv2
import numpy as np
from PIL import Image

# BGR colours (OpenCV is BGR, not RGB)
BLUE_BG = (110, 46, 10)      # deep blueprint blue  #0A2E6E
BLUE_GRID = (95, 40, 8)      # faint grid
BLUE_GRID_FINE = (85, 36, 7)
WHITE = (255, 255, 255)
LIGHT_BLUE = (255, 200, 120)  # inner parts
CYAN = (255, 220, 0)          # dimension lines / text (cyan in RGB)
DIM = (200, 200, 200)


def _find_object_rects(twin):
    """Every object in the twin -> list of (x, y, w, h) in px,
    largest first. The objects are assumed darker / different
    from the background (e.g. dark boards on white A4).

    Each contour is one object; internal parts (keys, traces)
    are inside the outer contour, so RETR_EXTERNAL keeps them
    from becoming objects of their own. A contour only counts
    if it is big enough and solid enough (fills most of its
    bounding box) — that skips shadows, smudges and noise."""
    gray = cv2.cvtColor(twin, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = gray.shape
    sheet_area = h * w
    rects = []
    for mask in (255 - bw, bw):  # dark objects, or light ones
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if not (0.008 * sheet_area < area < 0.95 * sheet_area):
                continue
            x, y, rw, rh = cv2.boundingRect(c)
            if rw < 10 or rh < 10:
                continue
            # solidity: skip wispy shadows/smudge (a real object
            # fills a decent part of its bbox — a guitar is ~0.5,
            # a board ~0.95; the tighten step below has its own
            # stricter bar, so this one stays low on purpose)
            if area < 0.3 * rw * rh:
                continue
            rects.append((x, y, rw, rh))
    # de-duplicate near-identical rects (same object found in
    # both masks), keeping the largest of a cluster
    rects.sort(key=lambda r: r[2] * r[3], reverse=True)
    kept = []
    for r in rects:
        if any(abs(r[0] - k[0]) < 8 and abs(r[1] - k[1]) < 8
               and abs(r[2] - k[2]) < 8 and abs(r[3] - k[3]) < 8 for k in kept):
            continue
        kept.append(r)
    return kept


def _tighten_rect(mask, rect):
    """Trim a bounding rect to the object's main body.

    A plug sticking out of a board, or pins past the edge, make
    the plain bounding box too big (an Uno read 84 mm tall
    instead of 69, because of the USB plug). The plug only fills
    a small part of that side, while the board's own edge fills
    nearly all of it — so shrink each side in until that side
    is more than half filled. Only for roughly rectangular
    objects (solidity > 0.8): a curved object like a guitar
    keeps its full box, neck and all."""
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
    """True if the rect's interior is mostly dark pixels."""
    gray = cv2.cvtColor(twin, cv2.COLOR_BGR2GRAY)
    return float(gray[y:y + h, x:x + w].mean()) < 128


def _find_object_rect(twin):
    """Largest object only (kept for callers that want one)."""
    rects = _find_object_rects(twin)
    return rects[0] if rects else None


def _inner_rects(twin):
    """Smaller rectangular parts inside the twin (calculator buttons,
    screen, etc.) -> list of (x, y, w, h) in px. Used for the inner
    line drawing. Best-effort: missed parts just don't get drawn."""
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
        # keep roughly rectangular parts
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) < 4:
            continue
        out.append((x, y, rw, rh))
    # de-duplicate near-identical rects (keep the largest of a cluster)
    out.sort(key=lambda r: r[2] * r[3], reverse=True)
    kept = []
    for r in out:
        if any(abs(r[0] - k[0]) < 6 and abs(r[1] - k[1]) < 6
               and abs(r[2] - k[2]) < 6 and abs(r[3] - k[3]) < 6 for k in kept):
            continue
        kept.append(r)
        if len(kept) >= 60:  # a calculator has ~40 keys; cap the drawing
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
    """Horizontal dimension line with arrows + label, in px coords."""
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
    """Vertical dimension line with arrows + label, in px coords."""
    cv2.line(img, (x, y0), (x, y1), color, 2, cv2.LINE_AA)
    cv2.arrowedLine(img, (x, y0), (x, y0 + 14), color, 2, tipLength=0.6,
                    line_type=cv2.LINE_AA)
    cv2.arrowedLine(img, (x, y1), (x, y1 - 14), color, 2, tipLength=0.6,
                    line_type=cv2.LINE_AA)
    cv2.line(img, (x - 6, y0), (x + 6, y0), color, 2, cv2.LINE_AA)
    cv2.line(img, (x - 6, y1), (x + 6, y1), color, 2, cv2.LINE_AA)
    # vertical text: draw it on a small canvas and rotate it
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
    """Blueprint sheet (BGR) -> PDF bytes that print at true 1:1 scale.

    The sheet is drawn at px_per_mm pixels per mm, so embedding it
    at px_per_mm * 25.4 dpi makes 1 px = 1/px_per_mm mm on paper:
    hold a printed object outline against the real object and they
    match. The PDF page is the full sheet (twin + margins + title
    block), so at 5 px/mm an A4 sheet comes out 242 x 333 mm — print
    it on A3, or tile it with the printer's poster mode on A4.
    No new dependency: Pillow ships the PDF writer."""
    rgb = cv2.cvtColor(sheet, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, "PDF", resolution=float(px_per_mm) * 25.4)
    return buf.getvalue()


def save_blueprint_pdf(sheet, px_per_mm, path):
    """Write a true-scale (1:1) PDF of the blueprint sheet to `path`."""
    with open(path, "wb") as f:
        f.write(blueprint_pdf_bytes(sheet, px_per_mm))


def make_blueprint(twin, width_mm, height_mm, px_per_mm,
                   title="CALCULATOR"):
    """twin (BGR) -> (blueprint sheet (BGR), info dict).

    The sheet is the twin plus margins: 90 px left/top for the
    dimension lines, 90 px bottom for the title block.
    """
    th, tw = twin.shape[:2]
    mx, my, bottom = 90, 90, 90
    sheet = np.zeros((th + my + bottom, tw + mx + 70, 3), dtype=np.uint8)
    sheet[:] = BLUE_BG
    ox, oy = mx, my  # twin origin inside the sheet

    # faint 10 mm grid over the whole sheet
    step = max(1, int(round(10 * px_per_mm)))
    for x in range(0, sheet.shape[1], step):
        cv2.line(sheet, (x, 0), (x, sheet.shape[0]), BLUE_GRID_FINE, 1)
    for y in range(0, sheet.shape[0], step):
        cv2.line(sheet, (0, y), (sheet.shape[1], y), BLUE_GRID_FINE, 1)
    # stronger 50 mm grid
    step5 = max(1, int(round(50 * px_per_mm)))
    for x in range(0, sheet.shape[1], step5):
        cv2.line(sheet, (x, 0), (x, sheet.shape[0]), BLUE_GRID, 1)
    for y in range(0, sheet.shape[0], step5):
        cv2.line(sheet, (0, y), (sheet.shape[1], y), BLUE_GRID, 1)

    # --- line drawing of the twin: edges in white ---
    # Canny 25/75, no dilation: the keys are only slightly
    # lighter than the black body (gradient ~40), so the
    # first build's Canny 50/150 found 0 edge pixels in the
    # key area and the key outlines were missing from the
    # drawing. At 25/75 the keys draw as thin lines — the
    # same edge tracing, just with the lines actually there.
    gray = cv2.cvtColor(twin, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 25, 75)
    sheet[oy:oy + th, ox:ox + tw][edges > 0] = WHITE

    # --- sheet frame: from the known twin bounds, not the
    # photo's edges (the photo's sheet border only ever draws
    # in fragments — top 59%, sides patchy — because it is
    # low-contrast; the twin IS the sheet, so its border is
    # exact and always complete) ---
    cv2.rectangle(sheet, (ox, oy), (ox + tw, oy + th), WHITE, 2)

    # --- every object: outline + its own dimensions ---
    # (before, only the largest blob got an outline and the
    # second board on the sheet was drawn but never measured)
    gray_obj = cv2.cvtColor(twin, cv2.COLOR_BGR2GRAY)
    gray_obj = cv2.GaussianBlur(gray_obj, (5, 5), 0)
    _, bw_obj = cv2.threshold(gray_obj, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    objects = _find_object_rects(twin)
    if not objects:
        objects = [(0, 0, tw, th)]  # fallback: whole twin is the object
    obj_infos = []
    for (x, y, w, h) in objects:
        # tighten: trim plugs/pins past the edge, but only when
        # the shape is roughly rectangular (solidity > 0.8)
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
        cv2.rectangle(sheet, (ox + x, oy + y), (ox + x + w, oy + y + h),
                      WHITE, 3, cv2.LINE_AA)
        obj_w_mm = w / px_per_mm
        obj_h_mm = h / px_per_mm
        obj_infos.append({"w_mm": round(obj_w_mm, 1),
                          "h_mm": round(obj_h_mm, 1),
                          "px": [int(x), int(y), int(w), int(h)]})
        cx, cy = ox + x + w // 2, oy + y + h // 2
        # dashed centre lines, just past the object
        _dashed_line(sheet, (ox + x - 30, cy), (ox + x + w + 30, cy), DIM)
        _dashed_line(sheet, (cx, oy + y - 30), (cx, oy + y + h + 30), DIM)
        # dimension lines, tight to the object: width above it,
        # height on its outer side (left half -> left, else right)
        dim_y = oy + y - 30
        cv2.line(sheet, (ox + x, oy + y), (ox + x, dim_y - 5), CYAN, 1)
        cv2.line(sheet, (ox + x + w, oy + y), (ox + x + w, dim_y - 5), CYAN, 1)
        _dim_h(sheet, ox + x, ox + x + w, dim_y, f"{obj_w_mm:.0f} mm")
        if cx < ox + tw // 2:
            dim_x = ox + x - 30
            cv2.line(sheet, (ox + x, oy + y), (dim_x + 5, oy + y), CYAN, 1)
            cv2.line(sheet, (ox + x, oy + y + h), (dim_x + 5, oy + y + h), CYAN, 1)
        else:
            dim_x = ox + x + w + 30
            cv2.line(sheet, (ox + x + w, oy + y), (dim_x - 5, oy + y), CYAN, 1)
            cv2.line(sheet, (ox + x + w, oy + y + h), (dim_x - 5, oy + y + h), CYAN, 1)
        _dim_v(sheet, oy + y, oy + y + h, dim_x, f"{obj_h_mm:.0f} mm")

    for (ix, iy, iw, ih) in _inner_rects(twin):
        cv2.rectangle(sheet, (ox + ix, oy + iy), (ox + ix + iw, oy + iy + ih),
                      LIGHT_BLUE, 1)

    # --- border + title block (game-blueprint style) ---
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
    cv2.putText(sheet,
                f"REF {width_mm:.0f} x {height_mm:.0f} mm  |  SCALE {px_per_mm:g} px/mm  |  "
                f"GRID 10 mm  |  {today}",
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
        "sheet_px": [int(sheet.shape[1]), int(sheet.shape[0])],
        "ref_w_mm": width_mm,
        "ref_h_mm": height_mm,
        "px_per_mm": px_per_mm,
    }
    return sheet, info
