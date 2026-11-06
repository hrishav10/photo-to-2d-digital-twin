# Photo to 2D Digital Twin — Blueprint

One photo + a known reference = a true top-down, measurable 2D twin
+ a dimensioned blueprint. Photo only — no live feed
for now (run with `--live` to re-enable the webcam mode).

No COLMAP, no GPU, no 3D. Just one homography warp, in the browser.

## What you get: real twin + blueprint

Every photo produces **two** twins, side by side, plus the photo:

- **① Your photo** — angled, as taken.
- **② Real twin** — the top-down photo, as it looks.
- **③ Blueprint** — the same twin redrawn as a technical drawing:
  deep-blue paper, white line drawing of the object and its parts,
  dimension lines with arrows labelled in **real mm**, dashed
  centre lines, and a title block with the name you type in —
  there is no default object name, the box starts blank.

Click any window to open it big (⛶ button, or click the photo /
twin). Close with ✕, Esc, or clicking outside.

The dimensions are real: the object's width/height are measured
from its outline in the twin, using the homography scale
(px / px_per_mm = mm).


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

## Curved objects — e.g. a guitar on the floor

You never trace the curve. The four clicks are always the
corners of a rectangle on the floor plane — the reference
(A4) or an imaginary bounding box around the object — never
the object's outline. The warp straightens the whole plane,
so the curve comes along for free and appears correctly in
the twin and the blueprint.

For a guitar lying flat on the floor, use the 30 cm scale
mode: click the four corners of a rectangle on the floor
that encloses the guitar (top of the headstock, both sides
of the body), then the two ends of the scale lying flat
beside it. The blueprint then measures the guitar's
bounding box — its full length and the body's width — and
its outline is traced from the largest contour in the twin.
Keep the guitar flat on the floor; anything sticking up
will look stretched.

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
| `blueprint.py` | Twin → blueprint sheet: blue paper, white outlines, dimension lines in mm, centre lines, title block (PNG, plus an SVG export) |
| `templates/index.html` | Three-window UI: photo, real twin, blueprint — click-to-expand lightbox |
| `photo_to_2d.py` | CLI version: one photo → top-down PNG (+ optional SVG plan in mm, `--blueprint` sheet) |
| `requirements.txt` | Flask, opencv-python, numpy |

## License

MIT — see [LICENSE](LICENSE).
