import argparse
import json
import os
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_file

from blueprint import blueprint_pdf_bytes, format_len, make_blueprint
from detect import detect_quad, detect_ruler

app = Flask(__name__)

CALIB_FILE = os.path.join(os.path.dirname(__file__), "calibration.json")
DEFAULT_WIDTH_MM = 210.0
DEFAULT_HEIGHT_MM = 297.0
DEFAULT_PX_PER_MM = 2.0
PHOTO_PX_PER_MM = 5.0

state = {
    "homography": None,
    "width_mm": DEFAULT_WIDTH_MM,
    "height_mm": DEFAULT_HEIGHT_MM,
    "px_per_mm": DEFAULT_PX_PER_MM,
    "show_grid": True,
    "fps": 0.0,
    "camera_ok": False,
    "camera_error": "",
}


def save_calibration():
    if state["homography"] is None:
        return
    with open(CALIB_FILE, "w") as f:
        json.dump({
            "homography": state["homography"].tolist(),
            "width_mm": state["width_mm"],
            "height_mm": state["height_mm"],
            "px_per_mm": state["px_per_mm"],
        }, f)


def load_calibration():
    if not os.path.exists(CALIB_FILE):
        return
    try:
        with open(CALIB_FILE) as f:
            d = json.load(f)
        state["homography"] = np.array(d["homography"], dtype=np.float32)
        state["width_mm"] = d["width_mm"]
        state["height_mm"] = d["height_mm"]
        state["px_per_mm"] = d.get("px_per_mm", DEFAULT_PX_PER_MM)
        print(f"Loaded saved calibration from {CALIB_FILE}")
    except Exception as e:
        print(f"Ignoring bad {CALIB_FILE}: {e}")


def compute_homography(src_points, width_mm, height_mm, px_per_mm):
    out_w = int(round(width_mm * px_per_mm))
    out_h = int(round(height_mm * px_per_mm))
    src = np.array(src_points, dtype=np.float32)
    dst = np.array([[0, 0], [out_w - 1, 0],
                    [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    h = cv2.getPerspectiveTransform(src, dst)
    return h, out_w, out_h


def twin_size():
    return (int(round(state["width_mm"] * state["px_per_mm"])),
            int(round(state["height_mm"] * state["px_per_mm"])))


photo_state = {
    "image": None,
    "filename": "",
    "homography": None,
    "width_mm": DEFAULT_WIDTH_MM,
    "height_mm": DEFAULT_HEIGHT_MM,
    "px_per_mm": PHOTO_PX_PER_MM,
    "title": "OBJECT",
    "calibration_mode": "a4",
    "ruler_twin": None,
    "unit": "mm",
}


def draw_grid(img, px_per_mm):
    step = int(100 * px_per_mm)
    if step <= 0:
        return img
    h, w = img.shape[:2]
    for x in range(0, w, step):
        cv2.line(img, (x, 0), (x, h), (0, 255, 0), 1)
    for y in range(0, h, step):
        cv2.line(img, (0, y), (w, y), (0, 255, 0), 1)
    return img


def detect_aruco_points(frame):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    try:
        detector = cv2.aruco.ArucoDetector(dictionary)
        corners_list, ids, _ = detector.detectMarkers(frame)
    except AttributeError:
        corners_list, ids, _ = cv2.aruco.detectMarkers(frame, dictionary)
    if ids is None or sorted(ids.flatten().tolist()) != [0, 1, 2, 3]:
        found = sorted(ids.flatten().tolist()) if ids is not None else []
        return None, found
    by_id = {int(i): c[0].mean(axis=0) for i, c in zip(ids.flatten(), corners_list)}
    return [by_id[i].tolist() for i in (0, 1, 2, 3)], [0, 1, 2, 3]


def warp_photo_to_twin():
    if photo_state["image"] is None or photo_state["homography"] is None:
        return None
    w = int(round(photo_state["width_mm"] * photo_state["px_per_mm"]))
    h = int(round(photo_state["height_mm"] * photo_state["px_per_mm"]))
    twin = cv2.warpPerspective(photo_state["image"],
                               photo_state["homography"], (w, h))
    if state["show_grid"]:
        twin = draw_grid(twin, photo_state["px_per_mm"])
    return twin


def warp_photo_to_blueprint():
    if photo_state["image"] is None or photo_state["homography"] is None:
        return None, None
    w = int(round(photo_state["width_mm"] * photo_state["px_per_mm"]))
    h = int(round(photo_state["height_mm"] * photo_state["px_per_mm"]))
    twin = cv2.warpPerspective(photo_state["image"],
                               photo_state["homography"], (w, h))
    exclude = None
    if (photo_state["calibration_mode"] == "scale"
            and photo_state["ruler_twin"]):
        half_w = int(round(20 * photo_state["px_per_mm"]))
        (x1, y1), (x2, y2) = photo_state["ruler_twin"]
        exclude = [((x1, y1), (x2, y2), half_w)]
    sheet, info = make_blueprint(
        twin, photo_state["width_mm"], photo_state["height_mm"],
        photo_state["px_per_mm"], title=photo_state["title"],
        exclude=exclude, unit=photo_state["unit"])
    return sheet, info


def calibrate_with_scale(corners, ruler, ruler_mm):
    import math
    src = np.array(corners, dtype=np.float32)
    def dist(a, b):
        return float(math.hypot(a[0] - b[0], a[1] - b[1]))
    out_w = int(round((dist(src[0], src[1]) + dist(src[3], src[2])) / 2))
    out_h = int(round((dist(src[0], src[3]) + dist(src[1], src[2])) / 2))
    dst = np.array([[0, 0], [out_w - 1, 0],
                    [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    h = cv2.getPerspectiveTransform(src, dst)
    ruler_pts = np.array(ruler, dtype=np.float32).reshape(-1, 1, 2)
    warped_ruler = cv2.perspectiveTransform(ruler_pts, h).reshape(-1, 2)
    ruler_warped_px = dist(warped_ruler[0], warped_ruler[1])
    if ruler_warped_px <= 0:
        raise ValueError("ruler endpoints are the same point")
    px_per_mm = ruler_warped_px / float(ruler_mm)
    return h, out_w, out_h, px_per_mm, out_w / px_per_mm, out_h / px_per_mm


def _set_photo_title(d):
    title = str(d.get("title") or "").strip()
    photo_state["title"] = title.upper()[:30] if title else "OBJECT"


def warp_to_twin(frame):
    if state["homography"] is None:
        out = frame.copy()
        cv2.putText(out, "NOT CALIBRATED", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(out, "Click 'Calibrate' or 'Auto ArUco' above",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return out
    w, h = twin_size()
    twin = cv2.warpPerspective(frame, state["homography"], (w, h))
    if state["show_grid"]:
        twin = draw_grid(twin, state["px_per_mm"])
    return twin


class Camera:

    def __init__(self, index=0):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.lock = threading.Lock()
        self.frame = None
        self.running = False
        if not self.cap.isOpened():
            state["camera_error"] = f"Cannot open camera {index}. Is it plugged in / in use by another app?"
            print(state["camera_error"])
            return
        state["camera_ok"] = True
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            with self.lock:
                self.frame = frame

    def get_frame(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        self.cap.release()


camera = None


def encode_jpeg(frame, quality=70):
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None


def mjpeg_stream(make_frame):
    while True:
        frame = camera.get_frame() if camera else None
        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Waiting for camera...",
                        (150, 240), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (255, 255, 255), 2)
        out = make_frame(frame)
        data = encode_jpeg(out)
        if data is None:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
        time.sleep(0.03)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(mjpeg_stream(lambda f: f),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/twin_feed")
def twin_feed():
    return Response(mjpeg_stream(warp_to_twin),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/snapshot.jpg")
def snapshot():
    frame = camera.get_frame() if camera else None
    if frame is None:
        return jsonify({"error": "no camera frame yet"}), 503
    return Response(encode_jpeg(frame, quality=85), mimetype="image/jpeg")


@app.route("/snapshot_twin.png")
def snapshot_twin():
    frame = camera.get_frame() if camera else None
    if frame is None or state["homography"] is None:
        return jsonify({"error": "not calibrated yet"}), 400
    twin = warp_to_twin(frame)
    twin = cv2.resize(twin, None, fx=2.5, fy=2.5,
                      interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".png", twin)
    if not ok:
        return jsonify({"error": "encode failed"}), 500
    return Response(buf.tobytes(), mimetype="image/png",
                    headers={"Content-Disposition":
                             "attachment; filename=twin_topdown.png"})


def _objects_message(info):
    unit = info.get("unit", "mm")
    objs = info.get("objects") or []
    if len(objs) > 1:
        parts = ", ".join(f"{format_len(o['w_mm'], unit)} x "
                          f"{format_len(o['h_mm'], unit)}" for o in objs)
        return f"Objects ({len(objs)}): {parts}. "
    return (f"Object: {format_len(info['object_w_mm'], unit)} x "
            f"{format_len(info['object_h_mm'], unit)}. ")


@app.route("/upload_photo", methods=["POST"])
def upload_photo():
    if "photo" not in request.files:
        return jsonify({"error": "no file — form field must be named 'photo'"}), 400
    f = request.files["photo"]
    if not f.filename:
        return jsonify({"error": "empty filename"}), 400
    data = np.frombuffer(f.read(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "could not decode that file as an image"}), 400
    h, w = img.shape[:2]
    if max(h, w) > 2000:
        scale = 2000 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    photo_state["image"] = img
    photo_state["filename"] = f.filename
    photo_state["homography"] = None
    photo_state["ruler_twin"] = None
    return jsonify({"ok": True, "filename": f.filename,
                    "width_px": int(img.shape[1]), "height_px": int(img.shape[0]),
                    "message": f"Uploaded {f.filename} ({img.shape[1]}x{img.shape[0]}). "
                               "Now click its 4 corners."})


@app.route("/uploaded.jpg")
def uploaded_jpg():
    if photo_state["image"] is None:
        return jsonify({"error": "no photo uploaded yet"}), 404
    return Response(encode_jpeg(photo_state["image"], quality=85),
                    mimetype="image/jpeg")


@app.route("/photo_twin.jpg")
def photo_twin_jpg():
    twin = warp_photo_to_twin()
    if twin is None:
        return jsonify({"error": "upload a photo and calibrate it first"}), 400
    return Response(encode_jpeg(twin, quality=85), mimetype="image/jpeg")


@app.route("/photo_twin.png")
def photo_twin_png():
    twin = warp_photo_to_twin()
    if twin is None:
        return jsonify({"error": "upload a photo and calibrate it first"}), 400
    ok, buf = cv2.imencode(".png", twin)
    if not ok:
        return jsonify({"error": "encode failed"}), 500
    return Response(buf.tobytes(), mimetype="image/png",
                    headers={"Content-Disposition":
                             "attachment; filename=twin_topdown.png"})


@app.route("/photo_blueprint.jpg")
def photo_blueprint_jpg():
    sheet, _info = warp_photo_to_blueprint()
    if sheet is None:
        return jsonify({"error": "upload a photo and calibrate it first"}), 400
    return Response(encode_jpeg(sheet, quality=95), mimetype="image/jpeg")


@app.route("/photo_blueprint.png")
def photo_blueprint_png():
    sheet, _info = warp_photo_to_blueprint()
    if sheet is None:
        return jsonify({"error": "upload a photo and calibrate it first"}), 400
    ok, buf = cv2.imencode(".png", sheet)
    if not ok:
        return jsonify({"error": "encode failed"}), 500
    return Response(buf.tobytes(), mimetype="image/png",
                    headers={"Content-Disposition":
                             "attachment; filename=twin_blueprint.png"})


@app.route("/photo_blueprint.pdf")
def photo_blueprint_pdf():
    sheet, info = warp_photo_to_blueprint()
    if sheet is None:
        return jsonify({"error": "upload a photo and calibrate it first"}), 400
    pdf = blueprint_pdf_bytes(sheet, info["px_per_mm"])
    return Response(pdf, mimetype="application/pdf",
                    headers={"Content-Disposition":
                             "attachment; filename=twin_blueprint_1to1.pdf"})


@app.route("/photo_blueprint.json")
def photo_blueprint_json():
    _sheet, info = warp_photo_to_blueprint()
    if info is None:
        return jsonify({"error": "upload a photo and calibrate it first"}), 400
    return jsonify(info)


@app.route("/twin_blueprint_feed")
def twin_blueprint_feed():
    def make_blueprint_frame(frame):
        twin = warp_to_twin(frame)
        if state["homography"] is None:
            return twin
        sheet, _info = make_blueprint(
            twin, state["width_mm"], state["height_mm"],
            state["px_per_mm"], title=photo_state["title"])
        return sheet
    return Response(mjpeg_stream(make_blueprint_frame),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/photo_calibrate", methods=["POST"])
def photo_calibrate():
    if photo_state["image"] is None:
        return jsonify({"error": "upload a photo first"}), 400
    d = request.get_json(force=True)
    points = d.get("points")
    if not points or len(points) != 4:
        return jsonify({"error": "need exactly 4 points: TL, TR, BR, BL"}), 400
    width_mm = float(d.get("width_mm", DEFAULT_WIDTH_MM))
    height_mm = float(d.get("height_mm", DEFAULT_HEIGHT_MM))
    px_per_mm = float(d.get("px_per_mm", PHOTO_PX_PER_MM))
    _set_photo_title(d)
    photo_state["calibration_mode"] = "a4"
    photo_state["ruler_twin"] = None
    h, out_w, out_h = compute_homography(points, width_mm, height_mm, px_per_mm)
    photo_state["homography"] = h
    photo_state["width_mm"] = width_mm
    photo_state["height_mm"] = height_mm
    photo_state["px_per_mm"] = px_per_mm
    _sheet, info = warp_photo_to_blueprint()
    obj_txt = _objects_message(info)
    return jsonify({"ok": True, "twin_px": [out_w, out_h],
                    "blueprint": info,
                    "message": f"Photo twin ready: {out_w}x{out_h} px = "
                               f"{width_mm:.0f}x{height_mm:.0f} mm. "
                               f"{obj_txt}"
                               "Measure: distance_px / "
                               f"{px_per_mm:g} = distance_mm"})


@app.route("/photo_autodetect", methods=["POST"])
def photo_autodetect():
    if photo_state["image"] is None:
        return jsonify({"error": "upload a photo first"}), 400
    d = request.get_json(silent=True) or {}
    mode = d.get("mode", "a4")
    img = photo_state["image"]
    quad = detect_quad(img, mode=mode)
    if quad is None:
        what = "sheet" if mode == "a4" else "object"
        return jsonify({"ok": False,
                        "message": f"Could not find the {what} — click the corners."})
    points = [[int(round(x)), int(round(y))] for x, y in quad]
    ruler = None
    if mode == "scale":
        r = detect_ruler(img, quad)
        if r is not None:
            ruler = [[int(round(x)), int(round(y))] for x, y in r]
            msg = "Object and scale detected — drag any corner to adjust, then Generate."
        else:
            msg = "Object detected — now click the 2 ends of the scale, then Generate."
    else:
        msg = "Sheet detected — drag any corner to adjust, then Generate."
    return jsonify({"ok": True, "points": points, "ruler": ruler,
                    "message": msg})


@app.route("/photo_aruco", methods=["POST"])
def photo_aruco():
    if photo_state["image"] is None:
        return jsonify({"error": "upload a photo first"}), 400
    d = request.get_json(silent=True) or {}
    width_mm = float(d.get("width_mm", DEFAULT_WIDTH_MM))
    height_mm = float(d.get("height_mm", DEFAULT_HEIGHT_MM))
    px_per_mm = float(d.get("px_per_mm", PHOTO_PX_PER_MM))
    _set_photo_title(d)
    points, found = detect_aruco_points(photo_state["image"])
    if points is None:
        return jsonify({"error": f"need ArUco ids 0-3 in the photo, found {found}"}), 400
    h, out_w, out_h = compute_homography(points, width_mm, height_mm, px_per_mm)
    photo_state["calibration_mode"] = "a4"
    photo_state["ruler_twin"] = None
    photo_state["homography"] = h
    photo_state["width_mm"] = width_mm
    photo_state["height_mm"] = height_mm
    photo_state["px_per_mm"] = px_per_mm
    return jsonify({"ok": True, "twin_px": [out_w, out_h],
                    "message": "ArUco calibration locked for this photo."})


@app.route("/photo_calibrate_scale", methods=["POST"])
def photo_calibrate_scale():
    if photo_state["image"] is None:
        return jsonify({"error": "upload a photo first"}), 400
    d = request.get_json(force=True)
    corners = d.get("corners")
    ruler = d.get("ruler")
    if not corners or len(corners) != 4:
        return jsonify({"error": "need exactly 4 object corners: TL, TR, BR, BL"}), 400
    if not ruler or len(ruler) != 2:
        return jsonify({"error": "need exactly 2 ruler ends: 0 cm and 30 cm marks"}), 400
    ruler_mm = float(d.get("ruler_mm", 300))
    _set_photo_title(d)
    try:
        h, out_w, out_h, px_per_mm, width_mm, height_mm = calibrate_with_scale(
            corners, ruler, ruler_mm)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    photo_state["homography"] = h
    photo_state["width_mm"] = width_mm
    photo_state["height_mm"] = height_mm
    photo_state["px_per_mm"] = px_per_mm
    photo_state["calibration_mode"] = "scale"
    ruler_pts = np.array(ruler, dtype=np.float32).reshape(-1, 1, 2)
    photo_state["ruler_twin"] = cv2.perspectiveTransform(
        ruler_pts, h).reshape(-1, 2).tolist()
    _sheet, info = warp_photo_to_blueprint()
    obj_txt = _objects_message(info)
    return jsonify({"ok": True, "twin_px": [out_w, out_h],
                    "px_per_mm": round(px_per_mm, 3),
                    "area_mm": [round(width_mm, 1), round(height_mm, 1)],
                    "blueprint": info,
                    "message": f"Scale calibration ready: area "
                               f"{width_mm:.0f} x {height_mm:.0f} mm, "
                               f"{obj_txt}"
                               f"({px_per_mm:.2f} px/mm from the "
                               f"{ruler_mm:.0f} mm scale)."})


@app.route("/photo_preview", methods=["POST"])
def photo_preview():
    if photo_state["image"] is None:
        return jsonify({"error": "upload a photo first"}), 400
    d = request.get_json(force=True)
    mode = d.get("mode", "a4")
    try:
        if mode == "scale":
            corners = d.get("corners")
            ruler = d.get("ruler")
            if not corners or len(corners) != 4 or not ruler or len(ruler) != 2:
                return jsonify({"error": "need 4 corners + 2 ruler ends"}), 400
            h, out_w, out_h, _ppm, _wm, _hm = calibrate_with_scale(
                corners, ruler, float(d.get("ruler_mm", 300)))
        else:
            points = d.get("points")
            if not points or len(points) != 4:
                return jsonify({"error": "need exactly 4 points"}), 400
            h, out_w, out_h = compute_homography(
                points, float(d.get("width_mm", DEFAULT_WIDTH_MM)),
                float(d.get("height_mm", DEFAULT_HEIGHT_MM)),
                float(d.get("px_per_mm", PHOTO_PX_PER_MM)))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    preview = cv2.warpPerspective(photo_state["image"], h, (out_w, out_h))
    if max(out_w, out_h) > 1200:
        scale = 1200 / max(out_w, out_h)
        preview = cv2.resize(preview, (int(out_w * scale), int(out_h * scale)),
                             interpolation=cv2.INTER_AREA)
    return Response(encode_jpeg(preview, quality=80), mimetype="image/jpeg")


@app.route("/status")
def status():
    photo_twin_px = None
    if photo_state["homography"] is not None:
        photo_twin_px = [int(round(photo_state["width_mm"] * photo_state["px_per_mm"])),
                         int(round(photo_state["height_mm"] * photo_state["px_per_mm"]))]
    return jsonify({
        "camera_ok": state["camera_ok"],
        "camera_error": state["camera_error"],
        "calibrated": state["homography"] is not None,
        "width_mm": state["width_mm"],
        "height_mm": state["height_mm"],
        "px_per_mm": state["px_per_mm"],
        "twin_px": twin_size() if state["homography"] is not None else None,
        "show_grid": state["show_grid"],
        "photo_uploaded": photo_state["image"] is not None,
        "photo_filename": photo_state["filename"],
        "photo_calibrated": photo_state["homography"] is not None,
        "photo_twin_px": photo_twin_px,
        "photo_title": photo_state["title"],
        "calibration_mode": photo_state["calibration_mode"],
    })


@app.route("/calibrate", methods=["POST"])
def calibrate():
    d = request.get_json(force=True)
    points = d.get("points")
    if not points or len(points) != 4:
        return jsonify({"error": "need exactly 4 points: TL, TR, BR, BL"}), 400
    width_mm = float(d.get("width_mm", DEFAULT_WIDTH_MM))
    height_mm = float(d.get("height_mm", DEFAULT_HEIGHT_MM))
    px_per_mm = float(d.get("px_per_mm", DEFAULT_PX_PER_MM))
    h, _, _ = compute_homography(points, width_mm, height_mm, px_per_mm)
    state["homography"] = h
    state["width_mm"] = width_mm
    state["height_mm"] = height_mm
    state["px_per_mm"] = px_per_mm
    save_calibration()
    return jsonify({"ok": True, "twin_px": twin_size(),
                    "message": f"Calibrated: twin is {twin_size()[0]}x{twin_size()[1]} px "
                               f"= {width_mm:.0f}x{height_mm:.0f} mm"})


@app.route("/calibrate_aruco", methods=["POST"])
def calibrate_aruco():
    frame = camera.get_frame() if camera else None
    if frame is None:
        return jsonify({"error": "no camera frame yet"}), 503
    d = request.get_json(silent=True) or {}
    width_mm = float(d.get("width_mm", DEFAULT_WIDTH_MM))
    height_mm = float(d.get("height_mm", DEFAULT_HEIGHT_MM))
    px_per_mm = float(d.get("px_per_mm", DEFAULT_PX_PER_MM))

    points, found = detect_aruco_points(frame)
    if points is None:
        return jsonify({"error": f"need ArUco ids 0-3 in frame, found {found}. "
                                 "Print 4 markers and place them at the corners."}), 400
    h, _, _ = compute_homography(points, width_mm, height_mm, px_per_mm)
    state["homography"] = h
    state["width_mm"] = width_mm
    state["height_mm"] = height_mm
    state["px_per_mm"] = px_per_mm
    save_calibration()
    return jsonify({"ok": True, "twin_px": twin_size(),
                    "message": "ArUco calibration locked."})


@app.route("/reset", methods=["POST"])
def reset():
    state["homography"] = None
    if os.path.exists(CALIB_FILE):
        os.remove(CALIB_FILE)
    return jsonify({"ok": True})


@app.route("/toggle_grid", methods=["POST"])
def toggle_grid():
    d = request.get_json(silent=True) or {}
    state["show_grid"] = bool(d.get("show_grid", not state["show_grid"]))
    return jsonify({"ok": True, "show_grid": state["show_grid"]})


@app.route("/set_unit", methods=["POST"])
def set_unit():
    d = request.get_json(silent=True) or {}
    unit = d.get("unit", "mm")
    if unit not in ("mm", "cm", "m"):
        return jsonify({"error": "unit must be mm, cm or m"}), 400
    photo_state["unit"] = unit
    return jsonify({"ok": True, "unit": unit})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Photo to 2D digital twin (Flask)")
    parser.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    parser.add_argument("--live", action="store_true",
                        help="enable the (paused) live webcam mode")
    parser.add_argument("--no-camera", action="store_true",
                        help="deprecated: live mode is off by default now")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    load_calibration()
    if args.live:
        camera = Camera(args.camera)
    print("Photo to 2D twin running at http://%s:%d" % (args.host, args.port))
    print("  Photo mode: upload a photo, click 4 corners, twin + blueprint appear")
    print("  Live mode is paused (run with --live to re-enable it)")
    app.run(host=args.host, port=args.port, threaded=True)
