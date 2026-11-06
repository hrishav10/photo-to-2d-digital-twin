"""Auto-detect the reference/object in a photo — no clicking corners.

Finds the largest clean 4-corner shape in the uploaded photo:
  - A4 mode: the A4 sheet (bright white on a darker background).
  - Scale mode: the object itself, plus the long thin ruler
    (its two short-edge midpoints are the 0 cm and 30 cm ends).

Corners are refined to sub-pixel accuracy with cornerSubPix, so
the auto-placed points land exactly on the real corners — more
accurate than clicking by hand, not just faster.

Method: two independent detectors, both cheap and offline.
  1. Canny edges -> contours -> approxPolyDP with 4 vertices.
  2. Otsu threshold (bright region, and its inverse for dark
     objects) -> contours -> approxPolyDP with 4 vertices.
A4 mode prefers the threshold result (the sheet is the big
bright region); scale mode takes the largest quad overall.
Quads that are too small, nearly the whole frame, or too skinny
are rejected. If nothing clean is found, returns None and the
UI falls back to clicking.
"""
import cv2
import numpy as np


def _order_points(pts):
    """4 points -> TL, TR, BR, BL (same order as the click flow)."""
    pts = np.array(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def _quads_from_contours(contours, shape, min_edge_ratio=0.15):
    h, w = shape[:2]
    img_area = float(h * w)
    quads = []
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:12]:
        area = cv2.contourArea(cnt)
        if area < 0.03 * img_area or area > 0.97 * img_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = None
        for eps in (0.02, 0.03, 0.05):
            cand = cv2.approxPolyDP(cnt, eps * peri, True)
            if len(cand) == 4:
                approx = cand
                break
        if approx is None:
            continue
        pts = _order_points(approx.reshape(4, 2))
        # reject skinny / degenerate quads: shortest edge must be
        # at least min_edge_ratio of the longest edge (rulers are
        # thin by nature, so the ruler search passes a much
        # smaller ratio)
        edges = [float(np.linalg.norm(pts[(i + 1) % 4] - pts[i]))
                 for i in range(4)]
        if min(edges) < min_edge_ratio * max(edges):
            continue
        quads.append((area, pts))
    return quads


def _quads_from_edges(gray, min_edge_ratio=0.15):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)
    return [("edges", a, p) for a, p in
            _quads_from_contours(contours, gray.shape, min_edge_ratio)]


def _quads_from_threshold(gray, min_edge_ratio=0.15):
    out = []
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    for mask in (bw, 255 - bw):  # bright sheet, or dark object
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)
        out.extend([("threshold", a, p) for a, p in
                    _quads_from_contours(contours, gray.shape, min_edge_ratio)])
    return out


def _refine_corners(gray, pts):
    """Sub-pixel refinement; falls back to the raw points on failure."""
    try:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    40, 0.001)
        refined = cv2.cornerSubPix(
            gray, pts.reshape(-1, 1, 2).astype(np.float32),
            (11, 11), (-1, -1), criteria)
        if refined is None:
            return pts
        return refined.reshape(4, 2).astype(np.float32)
    except cv2.error:
        return pts


def detect_quad(img, mode="a4"):
    """Largest clean quadrilateral -> ordered TL, TR, BR, BL
    (float32, in photo pixels), or None if nothing clean is found."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cands = []
    try:
        cands.extend(_quads_from_edges(gray))
    except cv2.error:
        pass
    try:
        cands.extend(_quads_from_threshold(gray))
    except cv2.error:
        pass
    if not cands:
        return None
    if mode == "a4":
        # the sheet is the big bright region: threshold hits first
        cands.sort(key=lambda c: (0 if c[0] == "threshold" else 1, -c[1]))
    else:
        cands.sort(key=lambda c: -c[1])
    pts = _refine_corners(gray, cands[0][2])
    h, w = gray.shape
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    return pts


def _contours(gray):
    """Raw outer contours from both detectors (edges + threshold)."""
    out = []
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)
    out.extend(contours)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    for mask in (bw, 255 - bw):
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)
        out.extend(contours)
    return out


def detect_ruler(img, object_quad):
    """Find the 30 cm scale: a long thin rectangle.
    Returns its 2 end midpoints (the 0 cm and 30 cm marks), or None.
    Skips anything as big as the object itself.

    Uses minAreaRect, not approxPolyDP: a ruler's outline is often
    too wobbly (double edges from its border) to collapse to 4
    points, but it is still a rectangle."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    img_area = float(h * w)
    obj_area = 0.0
    if object_quad is not None:
        obj_area = float(cv2.contourArea(object_quad.astype(np.float32)))
    for cnt in sorted(_contours(gray), key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(cnt)
        if area < 0.005 * img_area or area > 0.25 * img_area:
            continue
        if area >= 0.9 * obj_area:
            continue  # the object itself (or the sheet), not the ruler
        (_cx, _cy), (rw, rh), _angle = cv2.minAreaRect(cnt)
        if min(rw, rh) <= 0:
            continue
        long_e, short_e = max(rw, rh), min(rw, rh)
        if long_e < 4 * short_e:
            continue  # not thin enough to be a ruler
        if area < 0.7 * rw * rh:
            continue  # not a solid rectangle
        pts = _order_points(cv2.boxPoints(cv2.minAreaRect(cnt)))
        edges = [float(np.linalg.norm(pts[(i + 1) % 4] - pts[i]))
                 for i in range(4)]
        # the two shortest edges are the ruler's ends; their
        # midpoints are the 0 cm and 30 cm marks
        order = np.argsort(edges)
        ends = np.array([((pts[i] + pts[(i + 1) % 4]) / 2.0)
                         for i in order[:2]], dtype=np.float32)
        ends[:, 0] = np.clip(ends[:, 0], 0, w - 1)
        ends[:, 1] = np.clip(ends[:, 1], 0, h - 1)
        return ends
    return None
