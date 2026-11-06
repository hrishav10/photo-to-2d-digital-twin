# Photo to 2D Digital Twin

Upload one photo of a flat object with a known reference in frame.
The app produces a top-down, measurable twin and a dimensioned
blueprint of the same object.

Photo only. No live feed, no account, no cloud — photos are held
in memory on the local machine.

## Output

1. **Source photo** — as taken.
2. **Top-down twin** — the warped, top-down view at 5 px/mm.
3. **Blueprint** — a static technical drawing: blue sheet, white
   outlines, dimension lines labelled in millimetres, centre
   lines, and a title block with the object name you enter.
   There is no default object name; leave the field blank and
   the sheet is titled `OBJECT`.

Click any of the three panels to open it at full size. Close
with the Close button, Esc, or by clicking outside the image.

Dimensions are measured from the object's outline in the twin,
using the calibration scale (pixels ÷ px/mm = millimetres).

## Calibration

Two references are supported. Choose one before calibrating.

### A4 sheet — for small objects

Place an A4 sheet (210 × 297 mm) flat in frame, corners sharp,
even light. Click its four corners in order: top-left, top-right,
bottom-right, bottom-left.

### 30 cm scale — for larger objects

Place a 30 cm ruler flat beside the object, in the same plane,
fully in frame. Click the object's four corners first, then the
two ends of the scale: the 0 cm mark, then the 30 cm mark.

The scale sets the real px/mm — the ruler is warped with the
same homography and then measured — so dimensions remain in
real millimetres with no A4 sheet, at any size.

ArUco markers (ids 0–3) can be used instead of clicking, in
A4 mode.

## Run

```bash
pip install -r requirements.txt
python3 app.py
# open http://127.0.0.1:5000
```

Live webcam mode is currently disabled. Re-enable it with
`python3 app.py --live`.

## CLI

```bash
python3 photo_to_2d.py photo.jpg
# click four corners; output: twin_topdown.png
python3 photo_to_2d.py photo.jpg --corners 120,80 900,110 880,700 140,660 --svg plan.svg
python3 photo_to_2d.py photo.jpg --corners 120,80 900,110 880,700 140,660 --blueprint blueprint.png --title SUBMARINE
```

## How it works

```
Photo → Calibrate once → Warp → Twin + Blueprint
        A4 corners, or    cv2.warpPerspective
        object corners +  blueprint.py redraws
        30 cm scale ends  the twin as a sheet
```

The homography is a single 3 × 3 matrix. Computing it is the
fiddly part; applying it is cheap. In scale mode, the warp size
comes from the object corners in the photo and the px/mm comes
from the warped 30 cm ruler.

## Files

| File | Purpose |
| --- | --- |
| `app.py` | Flask app: photo upload, A4 and 30 cm scale calibration, twin and blueprint routes |
| `blueprint.py` | Twin to blueprint sheet (PNG) and blueprint (SVG): blue sheet, white outlines, dimension lines in mm, centre lines, title block |
| `templates/index.html` | Web interface: upload, calibration, three result panels, full-size view |
| `photo_to_2d.py` | Command-line version: one photo to a top-down PNG, optional SVG plan and blueprint sheet |
| `requirements.txt` | Flask, opencv-python, numpy |

## License

MIT — see [LICENSE](LICENSE).
