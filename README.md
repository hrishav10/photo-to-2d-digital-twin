# Photo to 2D Digital Twin — Blueprint

One photo + a known reference = a true top-down, measurable 2D twin
+ a dimensioned blueprint. Photo only — no live feed
for now (run with `--live` to re-enable the webcam mode).

No COLMAP, no GPU, no 3D. Just one homography warp, in the browser.

## What you get: real twin + blueprint

Every photo produces **two** twins, side by side, plus the photo:

- **① Your photo** — angled, as taken.
- **② Real twin** — the top-down photo, as it looks.
- **③ Blueprint** — the same twin redrawn as a blueprint:
  deep-blue paper, white line drawing of the object and its parts,
  dimension lines with arrows labelled in **real mm**, dashed
  centre lines, and a title block with the name you type in —
  there is no default object name, the box starts blank.

The top shows the workflow as three steps — Upload,
Calibrate, Generate — with the current step highlighted;
the steps are the actions too: click Upload to pick a
photo, Calibrate to open calibration, Generate to apply
it. The next step also pops up on its own — upload a
photo and Calibrate opens by itself; after Generate the
page scrolls back up to the results. Each card
has an expand button and a menu (Download / Expand). The
bottom bar shows the file name and the Export buttons
(PNG for the twin, Blueprint for the drawing). Click any
card image to open it big. Close with X, Esc, or clicking
outside.

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

## Selecting the area

The calibration canvas is full-width, and selection is
deliberately precise:

- As you click, each point is shown as a numbered dot —
  no connecting lines, just the points.
- Drag any point to move it. Backspace (or Undo) removes
  the last point.
- **Preview** warps the photo from the points as clicked,
  without saving anything. Check it, adjust the points,
  preview again, then Apply.

The layout is full-width with no side margins, and the
interface uses only restrained fade and rise animations.

## Inner parts are detected on the object, not the whole photo

Parts (screen, keys) are found by running edge detection
on the object crop only, at low thresholds (Canny 25/75).
The old version ran Canny 50/150 on the whole twin:
calculator keys are only slightly lighter than the black
body, so no key edge was ever found and the drawing came
out with scattered fragments instead of the key grid.
The whole body itself is also skipped as a part, so it no
longer gets drawn twice on top of the outline.

## Every object gets measured, not just the biggest

The blueprint finds *all* objects on the sheet
(`_find_object_rects` in `blueprint.py`): each one gets
its own white outline, its own dimension lines (width
above it, height on its outer side) and its own centre
lines. The title block notes the count ("2 OBJECTS").
The status message and the Blueprint card footer list
every object's size. Before, only the largest blob was
outlined and a second board on the sheet was drawn but
never measured.

Each outline is tightened to the board itself
(`_tighten_rect`): a side is shrunk in until it is more
than half filled, so a USB plug sticking out of the top
(or pins past the edge) no longer inflates the reading
— an Uno that read 84 mm tall now reads its true 69.
Tightening only applies to roughly rectangular shapes;
a curved object (guitar) keeps its full bounding box.

The sheet frame is drawn from the known twin bounds,
not from the photo's edges — the photo's sheet border
only ever drew in fragments (top 59%, sides patchy),
so the frame is now a complete rectangle, always.

## The edge drawing shows every line

The blueprint's line drawing runs Canny 25/75 with no
dilation. The keys are only slightly lighter than the
black body, so the first build's Canny 50/150 found no
key edges at all and the key outlines were missing. At
25/75 the keys draw as thin lines — the same edge
tracing, with the lines actually there. No rectangles
are added: the lines are the traced edges.

## Auto-detect: no clicking corners

Upload a photo and the corners are found for you
(`detect.py`, `POST /photo_autodetect`): the largest
clean quadrilateral in the photo — the A4 sheet in A4
mode, the object itself in scale mode — refined to
sub-pixel accuracy with `cornerSubPix`, so the placed
points land exactly on the real corners. In scale mode
it also finds the ruler (a long thin rectangle) and
places its two ends. The Calibrate panel opens with
the points already placed: drag any corner to adjust,
then Generate. If nothing clean is found, it says so
and you click the corners as before. An "Auto-detect"
button in the panel re-runs it any time. Tested on
synthetic perspective photos: sheet and object corners
within 0.6 px, ruler ends within 2.6 px.

## The outline is the measured rectangle

The blueprint outline is the object's measured bounding
rectangle (the same 81 x 168 mm as the dimension lines),
not a traced photo contour. Tracing the contour with
approxPolyDP turned a real photo's rounded corners, glare
and shadow into big diagonal cuts, so the outline came out
skewed and did not look like the object. The rectangle
cannot skew that way. Inner parts (screen, keys) are still
detected from the photo and drawn inside it, in white.


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
         30 cm scale ends     it as a JPEG drawing
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
| `app.py` | Flask app: photo upload, A4 + 30 cm scale calibration, blueprint JPG/PNG/JSON routes (live mode paused, `--live` to re-enable) |
| `blueprint.py` | Twin → blueprint sheet (first-build recipe): blue paper, white edge drawing, measured outline, dimension lines in mm, centre lines, title block |
| `templates/index.html` | Three-window UI: photo, real twin, blueprint — click-to-expand lightbox |
| `photo_to_2d.py` | CLI version: one photo → top-down PNG (+ optional SVG plan in mm, `--blueprint` sheet) |
| `requirements.txt` | Flask, opencv-python, numpy |

## License

MIT — see [LICENSE](LICENSE).
