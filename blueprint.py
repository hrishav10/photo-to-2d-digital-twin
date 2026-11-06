#!/usr/bin/env python3
"""Twin -> Technical Drawing: turn a top-down twin photo into a
technical drawing, like a submarine blueprint in a game.

  Input : warped top-down twin (BGR), its real size in mm, px/mm
  Output: blueprint sheet (BGR) — same twin, but drawn as:
    * deep-blue paper, faint 10 mm grid
    * white line drawing of the object (edges + inner parts)
    * dimension lines with arrows, labelled in real mm
    * dashed centre lines through the object
    * title block: name, TOP VIEW, scale, reference size

The dimensions are real, not decoration: they come from the
homography scale (px / px_per_mm = mm). The object's own width
and height are measured from its largest contour in the twin.

Usage:
    from blueprint import make_blueprint
    sheet, info = make_blueprint(twin, 210, 297, 5.0, title="CALCULATOR")
    # info = {"object_w_mm": ..., "object_h_mm": ..., "object_px": (x,y,w,h)}
"""
import datetime

import cv2
import numpy as np

# BGR colours (OpenCV is BGR, not RGB)
BLUE_BG = (110, 46, 10)      # deep blueprint blue  #0A2E6E
BLUE_GRID = (95, 40, 8)      # faint grid
BLUE_GRID_FINE = (85, 36, 7)
WHITE = (255, 255, 255)
LIGHT_BLUE = (255, 200, 120)  # inner parts
CYAN = (255, 220, 0)          # dimension lines / text (cyan in RGB)
DIM = (200, 200, 200)


def _find_object_rect(twin):
    """Largest contour in the twin -> bounding rect (x, y, w, h) in px,
    or None. The object is assumed darker / different from the
    background (e.g. a dark calculator on white A4)."""
    gray = cv2.cvtColor(twin, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Object could be dark-on-light or light-on-dark: take the
    # contour set whose largest piece is not the whole sheet.
    contours, _ = cv2.findContours(255 - bw, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    h, w = gray.shape
    sheet_area = h * w
    candidates = [c for c in contours
                  if 0.02 * sheet_area < cv2.contourArea(c) < 0.98 * sheet_area]
    if not candidates:
        return None
    cnt = max(candidates, key=cv2.contourArea)
    return cv2.boundingRect(cnt)


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
    gray = cv2.cvtColor(twin, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    # keep only strong, object-like edges: dilate slightly so lines read
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    sheet[oy:oy + th, ox:ox + tw][edges > 0] = WHITE

    # --- object outline + inner parts ---
    obj = _find_object_rect(twin)
    if obj is None:
        obj = (0, 0, tw, th)  # fallback: the whole twin is the object
    x, y, w, h = obj
    cv2.rectangle(sheet, (ox + x, oy + y), (ox + x + w, oy + y + h),
                  WHITE, 3, cv2.LINE_AA)
    for (ix, iy, iw, ih) in _inner_rects(twin):
        cv2.rectangle(sheet, (ox + ix, oy + iy), (ox + ix + iw, oy + iy + ih),
                      LIGHT_BLUE, 1)

    obj_w_mm = w / px_per_mm
    obj_h_mm = h / px_per_mm
    cx, cy = ox + x + w // 2, oy + y + h // 2

    # --- dashed centre lines ---
    _dashed_line(sheet, (ox + x - 30, cy), (ox + x + w + 30, cy), DIM)
    _dashed_line(sheet, (cx, oy + y - 30), (cx, oy + y + h + 30), DIM)

    # --- dimension lines (real mm, from the object rect) ---
    # extension lines from the object to the dimension lines
    cv2.line(sheet, (ox + x, oy + y), (ox + x, my - 55), CYAN, 1)
    cv2.line(sheet, (ox + x + w, oy + y), (ox + x + w, my - 55), CYAN, 1)
    cv2.line(sheet, (ox + x, oy + y), (mx - 55, oy + y), CYAN, 1)
    cv2.line(sheet, (ox + x, oy + y + h), (mx - 55, oy + y + h), CYAN, 1)
    _dim_h(sheet, ox + x, ox + x + w, my - 40, f"{obj_w_mm:.0f} mm")
    _dim_v(sheet, oy + y, oy + y + h, mx - 40, f"{obj_h_mm:.0f} mm")

    # --- border + title block (game-blueprint style) ---
    cv2.rectangle(sheet, (8, 8), (sheet.shape[1] - 9, sheet.shape[0] - 9),
                  WHITE, 2)
    ty = sheet.shape[0] - bottom + 12
    cv2.line(sheet, (8, ty - 8), (sheet.shape[1] - 9, ty - 8), WHITE, 2)
    cv2.putText(sheet, f"{title}  —  TOP VIEW", (24, ty + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2, cv2.LINE_AA)
    today = datetime.date.today().isoformat()
    cv2.putText(sheet,
                f"REF {width_mm:.0f} x {height_mm:.0f} mm  |  SCALE {px_per_mm:g} px/mm  |  "
                f"GRID 10 mm  |  {today}",
                (24, ty + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, CYAN, 1,
                cv2.LINE_AA)
    cv2.putText(sheet, "TECHNICAL DRAWING", (sheet.shape[1] - 340, ty + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, CYAN, 2, cv2.LINE_AA)

    info = {
        "object_w_mm": round(obj_w_mm, 1),
        "object_h_mm": round(obj_h_mm, 1),
        "object_px": [int(v) for v in obj],
        "sheet_px": [int(sheet.shape[1]), int(sheet.shape[0])],
        "ref_w_mm": width_mm,
        "ref_h_mm": height_mm,
        "px_per_mm": px_per_mm,
    }
    return sheet, info
