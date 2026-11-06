#!/usr/bin/env python3
"""Twin -> Blueprint: turn a top-down twin photo into a game-style
blueprint drawing, like a submarine blueprint in a game.

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
    sheet, info = make_blueprint(twin, 210, 297, 5.0, title="SUBMARINE")
    # info = {"object_w_mm": ..., "object_h_mm": ..., "object_px": (x,y,w,h)}
"""
import datetime
import html

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


def _inner_rects(twin, obj=None):
    """Smaller rectangular parts inside the object (calculator
    buttons, screen, etc.) -> list of (x, y, w, h) in twin px.
    Used for the inner line drawing. Best-effort: missed parts
    just don't get drawn.

    Two things that used to break this on a real photo:
    - Canny ran at 50/150 on the whole twin. Calculator keys
      are only slightly lighter than the black body (gradient
      ~40), so no key edge was ever found and the drawing came
      out with scattered fragments instead of the key grid.
      It now runs at 25/75, on the object crop only.
    - The whole body itself used to pass the size filter and
      get drawn as a "part" on top of the outline. A rect that
      is ~the whole object is now skipped.
    """
    if obj is None:
        obj = _find_object_rect(twin)
    if obj is None:
        return []
    ox, oy, ow, oh = obj
    crop = twin[oy:oy + oh, ox:ox + ow]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 25, 75)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 200 or area > 0.25 * ow * oh:
            continue
        x, y, rw, rh = cv2.boundingRect(c)
        if rw < 8 or rh < 8:
            continue
        # the whole body is not a part
        if rw > 0.9 * ow and rh > 0.9 * oh:
            continue
        # keep roughly rectangular parts
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) < 4:
            continue
        out.append((x + ox, y + oy, rw, rh))
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
                 (int(x1 + dx * e), int(y1 + dy * e)), color, thickness,
                 cv2.LINE_AA)
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


def make_blueprint_svg(twin, width_mm, height_mm, px_per_mm,
                       title="OBJECT"):
    """Same blueprint as make_blueprint(), but as interactive SVG.

    Every part is a <rect class="part"> with data-*
    attributes in real mm, so the browser can click / hover it,
    show its size, and measure between points. Groups are
    #bp-grid, #bp-drawing, #bp-dims, #bp-centre, #bp-title so the
    UI can toggle layers. Returns (svg_string, info_dict).
    """
    if not title:
        title = "OBJECT"
    th, tw = twin.shape[:2]
    mx, my, bottom = 90, 90, 90
    sw, sh = tw + mx + 70, th + my + bottom
    ox, oy = mx, my
    obj = _find_object_rect(twin)
    if obj is None:
        obj = (0, 0, tw, th)
    x, y, w, h = obj
    obj_w_mm = w / px_per_mm
    obj_h_mm = h / px_per_mm
    cx, cy = ox + x + w // 2, oy + y + h // 2
    today = datetime.date.today().isoformat()
    esc_title = html.escape(str(title).upper())

    def mm(v):
        return v / px_per_mm

    parts = []
    # Outline = the measured bounding rectangle, NOT the traced
    # photo contour. The contour trace (approxPolyDP) turns a
    # real photo's rounded corners, glare and shadow into big
    # diagonal cuts, so the outline came out skewed and did not
    # look like the object. The rect is the measured size
    # (81 x 168 mm for this calculator) and is always straight.
    parts.append(
        f'<rect class="part main" data-name="{esc_title}" '
        f'data-w-mm="{obj_w_mm:.1f}" data-h-mm="{obj_h_mm:.1f}" '
        f'data-x-mm="{mm(x):.1f}" data-y-mm="{mm(y):.1f}" '
        f'x="{ox + x}" y="{oy + y}" width="{w}" height="{h}" '
        f'fill="none" stroke="#fff" stroke-width="3"/>')
    for i, (ix, iy, iw, ih) in enumerate(_inner_rects(twin, obj)):
        parts.append(
            f'<rect class="part inner" data-name="Part {i + 1}" '
            f'data-w-mm="{mm(iw):.1f}" data-h-mm="{mm(ih):.1f}" '
            f'data-x-mm="{mm(ix):.1f}" data-y-mm="{mm(iy):.1f}" '
            f'x="{ox + ix}" y="{oy + iy}" width="{iw}" height="{ih}" '
            f'fill="rgba(255,255,255,0.08)" stroke="#fff" stroke-width="2"/>')

    grid = []
    step = max(1, int(round(10 * px_per_mm)))
    for gx in range(0, sw, step):
        grid.append(f'<line x1="{gx}" y1="0" x2="{gx}" y2="{sh}" stroke="#123a6e" stroke-width="1"/>')
    for gy in range(0, sh, step):
        grid.append(f'<line x1="0" y1="{gy}" x2="{sw}" y2="{gy}" stroke="#123a6e" stroke-width="1"/>')

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {sw} {sh}" width="100%" data-px-per-mm="{px_per_mm:g}" shape-rendering="geometricPrecision" text-rendering="geometricPrecision">
<defs><marker id="arr" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L9,3 L0,6 Z" fill="#00dcff"/></marker></defs>
<rect width="{sw}" height="{sh}" fill="#0a2e6e"/>
<g id="bp-grid">{''.join(grid)}</g>
<g id="bp-viewport">
<g id="bp-drawing">{''.join(parts)}</g>
<g id="bp-centre" stroke="#c8c8c8" stroke-width="1" stroke-dasharray="14 10">
<line x1="{ox + x - 30}" y1="{cy}" x2="{ox + x + w + 30}" y2="{cy}"/>
<line x1="{cx}" y1="{oy + y - 30}" x2="{cx}" y2="{oy + y + h + 30}"/>
</g>
<g id="bp-dims" stroke="#00dcff" fill="#00dcff" font-family="sans-serif" font-size="22">
<line x1="{ox + x}" y1="{my - 40}" x2="{ox + x + w}" y2="{my - 40}" stroke-width="2" marker-start="url(#arr)" marker-end="url(#arr)"/>
<line x1="{ox + x}" y1="{oy + y}" x2="{ox + x}" y2="{my - 55}" stroke-width="1"/>
<line x1="{ox + x + w}" y1="{oy + y}" x2="{ox + x + w}" y2="{my - 55}" stroke-width="1"/>
<text x="{(2 * ox + 2 * x + w) // 2}" y="{my - 48}" text-anchor="middle" stroke="none">{obj_w_mm:.0f} mm</text>
<line x1="{mx - 40}" y1="{oy + y}" x2="{mx - 40}" y2="{oy + y + h}" stroke-width="2" marker-start="url(#arr)" marker-end="url(#arr)"/>
<line x1="{ox + x}" y1="{oy + y}" x2="{mx - 55}" y2="{oy + y}" stroke-width="1"/>
<line x1="{ox + x}" y1="{oy + y + h}" x2="{mx - 55}" y2="{oy + y + h}" stroke-width="1"/>
<text x="{mx - 48}" y="{(2 * oy + 2 * y + h) // 2}" text-anchor="middle" stroke="none" transform="rotate(-90 {mx - 48} {(2 * oy + 2 * y + h) // 2})">{obj_h_mm:.0f} mm</text>
</g>
</g>
<g id="bp-title">
<rect x="8" y="8" width="{sw - 16}" height="{sh - 16}" fill="none" stroke="#fff" stroke-width="2"/>
<line x1="8" y1="{sh - bottom + 4}" x2="{sw - 8}" y2="{sh - bottom + 4}" stroke="#fff" stroke-width="2"/>
<text x="24" y="{sh - bottom + 34}" fill="#fff" font-family="sans-serif" font-size="26" font-weight="bold">{esc_title} — TOP VIEW</text>
<text x="24" y="{sh - bottom + 64}" fill="#00dcff" font-family="sans-serif" font-size="18">REF {width_mm:.0f} x {height_mm:.0f} mm  |  SCALE {px_per_mm:g} px/mm  |  GRID 10 mm  |  {today}</text>
<text x="{sw - 190}" y="{sh - bottom + 34}" fill="#00dcff" font-family="sans-serif" font-size="26" font-weight="bold">BLUEPRINT</text>
</g>
</svg>'''
    info = {
        "object_w_mm": round(obj_w_mm, 1),
        "object_h_mm": round(obj_h_mm, 1),
        "object_px": [int(v) for v in obj],
        "sheet_px": [int(sw), int(sh)],
        "ref_w_mm": width_mm,
        "ref_h_mm": height_mm,
        "px_per_mm": px_per_mm,
    }
    return svg, info


def make_blueprint(twin, width_mm, height_mm, px_per_mm,
                   title="OBJECT"):
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

    # --- object outline + inner parts ---
    # Outline = the measured bounding rectangle. The traced
    # contour polygon is NOT used here: on a real photo,
    # approxPolyDP turns rounded corners, glare and shadow
    # into big diagonal cuts, so the outline came out skewed
    # and did not look like the object. The rect is the
    # measured size and is always straight.
    obj = _find_object_rect(twin)
    if obj is None:
        obj = (0, 0, tw, th)  # fallback: the whole twin is the object
    x, y, w, h = obj
    cv2.rectangle(sheet, (ox + x, oy + y), (ox + x + w, oy + y + h),
                  WHITE, 3, cv2.LINE_AA)
    for (ix, iy, iw, ih) in _inner_rects(twin, obj):
        cv2.rectangle(sheet, (ox + ix, oy + iy), (ox + ix + iw, oy + iy + ih),
                      WHITE, 2, cv2.LINE_AA)

    obj_w_mm = w / px_per_mm
    obj_h_mm = h / px_per_mm
    cx, cy = ox + x + w // 2, oy + y + h // 2

    # --- dashed centre lines ---
    _dashed_line(sheet, (ox + x - 30, cy), (ox + x + w + 30, cy), DIM)
    _dashed_line(sheet, (cx, oy + y - 30), (cx, oy + y + h + 30), DIM)

    # --- dimension lines (real mm, from the object rect) ---
    # extension lines from the object to the dimension lines
    cv2.line(sheet, (ox + x, oy + y), (ox + x, my - 55), CYAN, 1, cv2.LINE_AA)
    cv2.line(sheet, (ox + x + w, oy + y), (ox + x + w, my - 55), CYAN, 1, cv2.LINE_AA)
    cv2.line(sheet, (ox + x, oy + y), (mx - 55, oy + y), CYAN, 1, cv2.LINE_AA)
    cv2.line(sheet, (ox + x, oy + y + h), (mx - 55, oy + y + h), CYAN, 1, cv2.LINE_AA)
    _dim_h(sheet, ox + x, ox + x + w, my - 40, f"{obj_w_mm:.0f} mm")
    _dim_v(sheet, oy + y, oy + y + h, mx - 40, f"{obj_h_mm:.0f} mm")

    # --- border + title block (game-blueprint style) ---
    cv2.rectangle(sheet, (8, 8), (sheet.shape[1] - 9, sheet.shape[0] - 9),
                  WHITE, 2, cv2.LINE_AA)
    ty = sheet.shape[0] - bottom + 12
    cv2.line(sheet, (8, ty - 8), (sheet.shape[1] - 9, ty - 8), WHITE, 2,
             cv2.LINE_AA)
    cv2.putText(sheet, f"{title}  —  TOP VIEW", (24, ty + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2, cv2.LINE_AA)
    today = datetime.date.today().isoformat()
    cv2.putText(sheet,
                f"REF {width_mm:.0f} x {height_mm:.0f} mm  |  SCALE {px_per_mm:g} px/mm  |  "
                f"GRID 10 mm  |  {today}",
                (24, ty + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, CYAN, 1,
                cv2.LINE_AA)
    cv2.putText(sheet, "BLUEPRINT", (sheet.shape[1] - 190, ty + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, CYAN, 2, cv2.LINE_AA)

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
