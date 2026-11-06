# Photo to 2D Digital Twin

One photo + one A4 sheet in frame = a true top-down, measurable 2D twin.
Or use your webcam and get the twin live, updating in the other window.

No COLMAP, no GPU, no 3D. Just one homography warp, in the browser.

![photo → twin](https://img.shields.io/badge/python-3.9%2B-blue) ![flask](https://img.shields.io/badge/flask-3.0-black) ![opencv](https://img.shields.io/badge/opencv-4.8-green)

## What it does

```
Photo (angled)                    2D twin (top-down, measurable)
┌──────────────┐    click 4       ┌──────────────┐
│  ╱╲           │    corners  ──►  │              │
│ ╱  ╲  A4      │    of the A4     │  ┌────────┐  │
│╱____╲         │    sheet         │  │        │  │
└──────────────┘                  └──────────────┘
  1 photo, 1 A4 sheet              pixels ÷ 5 = mm
```

The twin is the flat, top-down view of the area inside your 4
reference corners — a floor, a tabletop, a field. It is *not* a 3D
model and it does not understand height: anything sticking up will
look stretched. That is the honest limit of a single-camera 2D twin,
and it is also why it is so cheap to run.

## Two ways to use it

### A. Photo mode — the original idea (no camera needed)

1. Take one photo with an **A4 sheet (210 × 297 mm) flat in frame**,
   corners sharp, even light.
2. Run the app, upload the photo.
3. Click the A4 corners in order: **top-left, top-right,
   bottom-right, bottom-left** → Apply.
4. The twin appears instantly in the other window at **5 px/mm**
   (A4 = 1050 × 1485 px). Measure: `distance_px / 5 = distance_mm`.

Or skip clicking: print 4 ArUco markers (ids 0–3), place them at the
corners, and hit **Auto: ArUco markers**.

### B. Live mode — webcam, twin updates instantly

1. Put the A4 sheet / markers in view of your webcam.
2. Calibrate once (same 4 clicks, or ArUco).
3. Every frame is warped with that same homography, so the right
   window is a live top-down twin at 2 px/mm. Move an object and
   watch it move in the twin.

### C. CLI — no web UI at all

```bash
python3 photo_to_2d.py room.jpg
# click 4 corners, output: twin_topdown.png
python3 photo_to_2d.py room.jpg --corners 120,80 900,110 880,700 140,660 --svg plan.svg
python3 photo_to_2d.py room.jpg --aruco
```

## Run it

```bash
pip install -r requirements.txt
python3 app.py
# open http://127.0.0.1:5000
```

No camera? Photo mode still works:

```bash
python3 app.py --no-camera
python3 app.py --camera 1 --port 5000   # second webcam / other port
```

## How it works

```
Capture ──► Calibrate once ──► Warp every image
(photo or      4 corners          cv2.warpPerspective
 webcam)      → homography H      (image, H) = 2D twin
```

- **Calibrate once, warp forever.** The homography is a single 3×3
  matrix. Computing it is the fiddly part; applying it is cheap
  (~a few ms), so it runs on every live frame.
- **One shared frame (live).** A background thread keeps only the
  newest webcam frame. Both MJPEG streams read that same frame,
  so the two windows never drift apart.
- **Measurable.** Twin size = reference size × px/mm, with a
  100 mm grid drawn on it, so it reads as a plan, not a photo.

## For a whole room instead of A4

The twin only covers the area inside the 4 corners. For a room,
place the 4 markers at the room corners and set Width/Height to the
room size in mm (e.g. 4200 × 3000) before calibrating. Same code,
bigger reference.

## Files

| File | What it does |
| --- | --- |
| `app.py` | Flask app: photo upload + live camera, homography, MJPEG streams, calibrate / ArUco / snapshot routes |
| `templates/index.html` | Two-window UI: photo mode on top, live mode below, click-to-calibrate |
| `photo_to_2d.py` | CLI version: one photo → top-down PNG (+ optional SVG plan in mm) |
| `requirements.txt` | Flask, opencv-python, numpy |

## Next steps

- **Live sensor layer:** colour zones of the twin from a sensor
  (ESP32 → MQTT) — the twin becomes a live dashboard.
- **Object tracking:** detect a coloured object / ArUco tag, plot
  its (x, y) in mm on the twin — a moving dot on the plan.
- **Vector plan:** trace the twin's largest contour to SVG in mm,
  like `photo_to_2d.py --svg`.

## License

MIT — see [LICENSE](LICENSE).
