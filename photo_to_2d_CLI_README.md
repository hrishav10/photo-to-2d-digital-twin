# Photo → 2D twin — starter

One phone photo + one A4 sheet (210 × 297 mm) in frame = a true top-down,
measurable 2D view. No COLMAP, no GPU, runs on the ThinkPad X240.

## Install (once, on the ThinkPad)

```bash
pip install opencv-python numpy
```

## Run

```bash
python3 photo_to_2d.py photo.jpg
# click the 4 corners of the A4 sheet: top-left, top-right, bottom-right, bottom-left
# output: twin_topdown.png (1050 × 1485 px at 5 px/mm)
```

Non-interactive (corners already known), and the live-twin extras:

```bash
python3 photo_to_2d.py photo.jpg --corners 120,80 900,110 880,700 140,660 --svg plan.svg
python3 photo_to_2d.py photo.jpg --aruco   # 4 printed ArUco markers, ids 0-3 = TL TR BR BL
```

## Measure

In the output, `distance in pixels / 5 = distance in mm`
(at the default 5 px/mm; `--px-per-mm 10` gives A4 at 2100 × 2970).

## Tips that decide quality

- Flat scene only: a floor, tabletop, or field. Height distorts.
- Even, overcast light; no harsh shadows (same rule as the guitar shoot).
- A4 sheet fully in frame, corners sharp — the warp is only as good as the corners.
- `--svg` traces the largest contour into mm units for a vector plan;
  the live sensor layer (ESP32 → MQTT → colour the SVG zones) is the next
  step, not included here.

## Verified

Tested 2026-10-30 in this workspace with a synthetic perspective photo:
output was exactly 1050 × 1485 px for A4 at 5 px/mm, with an SVG written.
End-to-end run on a real photo still needs the user to run it on their
ThinkPad (this VM has no camera/display for the click mode).
