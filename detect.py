import cv2
import numpy as np


def _order_points(pts):
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
    for mask in (bw, 255 - bw):
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)
        out.extend([("threshold", a, p) for a, p in
                    _quads_from_contours(contours, gray.shape, min_edge_ratio)])
    return out


def _refine_corners(gray, pts):
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
        cands.sort(key=lambda c: (0 if c[0] == "threshold" else 1, -c[1]))
    else:
        cands.sort(key=lambda c: -c[1])
    pts = _refine_corners(gray, cands[0][2])
    h, w = gray.shape
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    return pts


def _contours(gray):
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
            continue
        (_cx, _cy), (rw, rh), _angle = cv2.minAreaRect(cnt)
        if min(rw, rh) <= 0:
            continue
        long_e, short_e = max(rw, rh), min(rw, rh)
        if long_e < 4 * short_e:
            continue
        if area < 0.7 * rw * rh:
            continue
        pts = _order_points(cv2.boxPoints(cv2.minAreaRect(cnt)))
        edges = [float(np.linalg.norm(pts[(i + 1) % 4] - pts[i]))
                 for i in range(4)]
        order = np.argsort(edges)
        ends = np.array([((pts[i] + pts[(i + 1) % 4]) / 2.0)
                         for i in order[:2]], dtype=np.float32)
        ends[:, 0] = np.clip(ends[:, 0], 0, w - 1)
        ends[:, 1] = np.clip(ends[:, 1], 0, h - 1)
        return ends
    return None
