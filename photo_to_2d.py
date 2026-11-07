#!/usr/bin/env python3
"""Photo -> 2D twin: warp one angled photo into a true top-down, scaled view.

The pipeline from today's chat, in code:
  1. pick the 4 corners of a known-size reference in the photo
     (an A4 sheet, 210 x 297 mm, is the zero-cost default)
  2. homography-warp those corners into a rectangle
  3. save the top-down view, scaled so 1 mm = --px-per-mm pixels

Modes for step 1 (in order of preference):
  --corners x1,y1 x2,y2 x3,y3 x4,y4   non-interactive, TL TR BR BL
  --aruco                              auto-detect 4 printed ArUco markers
                                       (ids 0,1,2,3 placed TL TR BR BL)
  (default)                            click the 4 corners in the window,
                                       TL TR BR BL, then press any key

Optional --svg writes a simple SVG outline (largest contour of the warped
view) in real mm units, so the result is a measurable plan, not just a picture.

Optional --blueprint writes a game-style blueprint sheet: the twin
redrawn in white on blueprint-blue, with dimension lines labelled
in real mm, centre lines and a title block. Use --title to name it,
e.g. --title CALCULATOR.

Needs: pip install opencv-python numpy
Runs fine on a ThinkPad X240-class machine: this is one warp, not COLMAP.

Example:
  python3 photo_to_2d.py room.jpg
  python3 photo_to_2d.py room.jpg --corners 120,80 900,110 880,700 140,660
  python3 photo_to_2d.py room.jpg --aruco --px-per-mm 10
  python3 photo_to_2d.py calculator.jpg --corners 120,80 900,110 880,700 140,660 --blueprint calculator_blueprint.png --title CALCULATOR
"""
import argparse
import sys

import cv2
import numpy as np

from blueprint import make_blueprint, save_blueprint_pdf


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("photo", help="input photo (e.g. room.jpg)")
    p.add_argument("-o", "--output", default="twin_topdown.png",
                   help="output top-down image (default: twin_topdown.png)")
    p.add_argument("--width-mm", type=float, default=210.0,
                   help="reference width in mm (default: 210, an A4 sheet)")
    p.add_argument("--height-mm", type=float, default=297.0,
                   help="reference height in mm (default: 297, an A4 sheet)")
    p.add_argument("--px-per-mm", type=float, default=5.0,
                   help="output resolution (default: 5 px/mm -> A4 = 1050x1485)")
    p.add_argument("--corners", nargs=4, metavar="X,Y",
                   help="4 corners TL TR BR BL, e.g. 120,80 900,110 880,700 140,660")
    p.add_argument("--aruco", action="store_true",
                   help="detect 4 ArUco markers (ids 0-3 = TL TR BR BL) instead of clicking")
    p.add_argument("--svg", default=None, metavar="OUT.SVG",
                   help="also write an SVG outline in mm units")
    p.add_argument("--blueprint", default=None, metavar="OUT.PNG",
                   help="also write a game-style blueprint sheet (blue, dimensioned)")
    p.add_argument("--blueprint-pdf", default=None, metavar="OUT.PDF",
                   help="also write the blueprint sheet as a PDF that prints "
                        "at true 1:1 scale (hold it against the real object)")
    p.add_argument("--title", default="OBJECT",
                   help="title on the blueprint sheet (default: OBJECT)")
    return p.parse_args()


def parse_corners(values):
    pts = []
    for v in values:
        try:
            x, y = v.split(",")
            pts.append((float(x), float(y)))
        except ValueError:
            sys.exit(f"bad corner {v!r}: use X,Y (e.g. 120,80)")
    return np.array(pts, dtype=np.float32)


def click_corners(img):
    pts = []

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((float(x), float(y)))
            cv2.circle(img, (x, y), 8, (0, 0, 255), -1)
            cv2.putText(img, str(len(pts)), (x + 12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("click TL, TR, BR, BL - then any key", img)

    cv2.imshow("click TL, TR, BR, BL - then any key", img)
    cv2.setMouseCallback("click TL, TR, BR, BL - then any key", on_mouse)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if len(pts) != 4:
        sys.exit(f"need exactly 4 clicks, got {len(pts)}")
    return np.array(pts, dtype=np.float32)


def aruco_corners(img):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    try:
        detector = cv2.aruco.ArucoDetector(dictionary)
        corners_list, ids, _ = detector.detectMarkers(img)
    except AttributeError:  # older OpenCV
        corners_list, ids, _ = cv2.aruco.detectMarkers(img, dictionary)
    if ids is None or len(ids) != 4:
        found = sorted(ids.flatten().tolist()) if ids is not None else []
        sys.exit(f"ArUco: need exactly 4 markers (ids 0-3), found {found}")
    by_id = {int(i): c[0].mean(axis=0) for i, c in zip(ids.flatten(), corners_list)}
    if sorted(by_id) != [0, 1, 2, 3]:
        sys.exit(f"ArUco: markers must be ids 0,1,2,3, found {sorted(by_id)}")
    return np.array([by_id[i] for i in (0, 1, 2, 3)], dtype=np.float32)


def write_svg(warped, path, px_per_mm):
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(255 - bw, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("SVG: no contour found, skipping")
        return
    cnt = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
    pts = approx.reshape(-1, 2) / px_per_mm  # pixels -> mm
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    h_mm = warped.shape[0] / px_per_mm
    w_mm = warped.shape[1] / px_per_mm
    with open(path, "w") as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w_mm:.0f}mm" '
                f'height="{h_mm:.0f}mm" viewBox="0 0 {w_mm:.1f} {h_mm:.1f}">\n'
                f'  <polygon points="{poly}" fill="none" stroke="black"/>\n</svg>\n')
    print(f"SVG outline -> {path} ({len(pts)} points, mm units)")


def main():
    args = parse_args()
    img = cv2.imread(args.photo)
    if img is None:
        sys.exit(f"cannot read {args.photo!r}")

    if args.corners:
        src = parse_corners(args.corners)
    elif args.aruco:
        src = aruco_corners(img)
    else:
        print("Click the 4 corners: top-left, top-right, bottom-right, bottom-left")
        src = click_corners(img.copy())

    out_w = int(round(args.width_mm * args.px_per_mm))
    out_h = int(round(args.height_mm * args.px_per_mm))
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
                   dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, matrix, (out_w, out_h))
    cv2.imwrite(args.output, warped)
    print(f"top-down view -> {args.output} ({out_w}x{out_h} px = "
          f"{args.width_mm:.0f}x{args.height_mm:.0f} mm at {args.px_per_mm:g} px/mm)")
    print("Measure in the output: distance_px / "
          f"{args.px_per_mm:g} = distance_mm")
    if args.svg:
        write_svg(warped, args.svg, args.px_per_mm)
    if args.blueprint or args.blueprint_pdf:
        sheet, info = make_blueprint(warped, args.width_mm, args.height_mm,
                                     args.px_per_mm, title=args.title)
        if args.blueprint:
            cv2.imwrite(args.blueprint, sheet)
            print(f"Blueprint -> {args.blueprint} ({sheet.shape[1]}x{sheet.shape[0]} px, "
                  f"object {info['object_w_mm']} x {info['object_h_mm']} mm)")
        if args.blueprint_pdf:
            save_blueprint_pdf(sheet, args.px_per_mm, args.blueprint_pdf)
            print(f"Blueprint PDF (1:1 print) -> {args.blueprint_pdf} "
                  f"({sheet.shape[1] / args.px_per_mm:.0f} x "
                  f"{sheet.shape[0] / args.px_per_mm:.0f} mm sheet)")


if __name__ == "__main__":
    main()
