# Photo to 2D Digital Twin — Blueprint

One photo + a known reference = a true top-down, measurable 2D twin
+ an interactive, game-style blueprint. Photo only — no live feed
for now (run with `--live` to re-enable the webcam mode).

No COLMAP, no GPU, no 3D. Just one homography warp, in the browser.

## What you get: real twin + interactive blueprint

Every photo produces **two** twins, side by side, plus the photo:

- **① Your photo** — angled, as taken.
- **② Real twin** — the top-down photo, as it looks.
- **③ Blueprint** — the same twin redrawn like a game blueprint
  (a submarine blueprint, for example): deep-blue paper, white
  line drawing of the object and its parts, dimension lines with
  arrows labelled in **real mm**, dashed centre lines, and a title
  block with the name *you* type in — there is no default object
  name, the box starts blank (e.g. type `SUBMARINE`).

Click any window to open it big (⛶ button, or click the photo /
twin). Close with ✕, Esc, or clicking outside.

The dimensions are real: the object's width/height are measured
from its outline in the twin, using the homography scale
(px / px_per_mm = mm).

### Interactive blueprint — like in a game

- **Click a part** (the outline, a button, the screen) — it
  highlights yellow and the panel below shows its size in mm,
  its x/y position, and its area.
- **Hover a part** — it highlights.
- **Zoom / pan** — mouse wheel to zoom, drag to pan, or the
  ＋ / － / Reset buttons.
- **📏 Measure** — turn it on, click any two points on the
  blueprint, and it tells you the distance between them in mm.
- **Layers** — toggle Grid, Dimensions, Centre lines, Title block.

Download it as PNG or as SVG (**Download blueprint SVG** — the SVG
keeps the clickable parts).

## Two references: A4, or a 30 cm scale for big objects

### A. Small object — A4 sheet (210 × 297 mm)

1. Put an A4 sheet flat in frame, corners sharp, even light.
2. Upload the photo, hit **Calibrate**, click the A4 corners in
   order: **top-left, top-right, bottom-right, bottom-left** → Apply.
3. Twin + blueprint appear at **5 px/mm** (A4 = 1050 × 1485 px).

### B. Big object — 30 cm scale beside it

Won't fit on A4? Don't force it. Put a **30 cm ruler / scale**
flat beside the object, in the same plane, fully in frame.

1. Upload the photo, hit **Calibrate**, pick **30 cm scale**.
2. Click the object's **4 corners** first (red dots 1–4).
3. Then click the scale's **2 ends** — the 0 cm mark, then the
   30 cm mark (cyan dots 5–6) → Apply.
4. The scale sets the real px/mm (the ruler is warped with the
   same homography, then measured), so the blueprint dimensions
   are still in real mm — no A4 needed, any size works.

Or skip clicking: print 4 ArUco markers (ids 0–3), place them at
the corners, and use the A4 mode's auto-calibration.

## Run it

```bash
pip install -r requirements.txt
python3 app.py
# open http://127.0.0.1:5000
```

Live webcam mode is paused; re-enable it with `python3 app.py --live`.

## CLI — no web UI at all

```bash
python3 photo_to_2d.py room.jpg
# click 4 corners, output: twin_topdown.png
python3 photo_to_2d.py room.jpg --corners 120,80 900,110 880,700 140,660 --svg plan.svg
python3 photo_to_2d.py calculator.jpg --corners 120,80 900,110 880,700 140,660 --blueprint calculator_blueprint.png --title CALCULATOR
```

## How it works

```
Photo ──► Calibrate once ──► Warp ──► Real twin + Blueprint
         A4 corners, or       cv2.warpPerspective
         object corners +     blueprint.py redraws
         30 cm scale ends     it as an SVG/PNG
```

- **Calibrate once, warp forever.** The homography is a single 3×3
  matrix. Computing it is the fiddly part; applying it is cheap.
- **Measurable.** Twin size = reference size × px/mm, with a
  100 mm grid drawn on the real twin, so it reads as a plan.
- **Scale mode:** the warp size comes from the object corners in
  the photo; the px/mm comes from the warped 30 cm ruler.

## Files

| File | What it does |
| --- | --- |
| `app.py` | Flask app: photo upload, A4 + 30 cm scale calibration, blueprint SVG/PNG/JSON routes (live mode paused, `--live` to re-enable) |
| `blueprint.py` | Twin → blueprint sheet (PNG) and interactive blueprint (SVG): blue paper, white outlines, dimension lines in mm, centre lines, title block |
| `templates/index.html` | Three-window UI: photo, real twin, interactive blueprint — click-to-expand lightbox, zoom/pan, measure, layer toggles |
| `photo_to_2d.py` | CLI version: one photo → top-down PNG (+ optional SVG plan in mm, `--blueprint` sheet) |
| `requirements.txt` | Flask, opencv-python, numpy |

## License

MIT — see [LICENSE](LICENSE).
