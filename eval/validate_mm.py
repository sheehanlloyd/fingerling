"""Measurement accuracy against objects whose size I actually know.

This is the only thing in the repo that produces a defensible millimetre error
figure. Everything else either tests the maths against itself (tests/) or reports
what a model predicted (eval/train.py). Here a real camera looks at real objects
of known size, and the number that comes out is the number I'd quote.

The method:

  1. A credit card is in frame. ISO/IEC 7810 ID-1 is 85.60 x 53.98 mm worldwide,
     so its size is known to a hundredth of a millimetre without a caliper.
     Calibrate on it.
  2. Other objects of known size are in the same frame, lying flat.
  3. Map each object's outline into the board plane, measure it in millimetres,
     compare to the truth table.

Three things I want to be honest about before anyone reads a number off this.

FIRST, THE COINS ARE NOT ON THE CALIBRATION PLANE. A card is about 0.76 mm thick
and a coin about 1.75 mm, so a coin's top face sits roughly 1 mm nearer the
camera than the card's. That's a magnification error of about h/D, a tenth of a
percent at arm's length. It is a BIAS, not noise, so averaging more shots will
not remove it. Pass --working-distance-mm and this prints what that bias should
be, next to the bias actually measured. If those two disagree wildly, something
other than coplanarity is wrong.

SECOND, MATCHING IS BY RANK, NOT BY NEAREST VALUE. Given three expected objects
and three detections, I sort both by size and pair them in order. Nearest-value
matching would assign every detection to whichever truth it's closest to, which
drags the reported error toward zero by construction, because a broken measurement would
look good. Rank matching can still be wrong if two objects are close enough in
size to swap, so the summary says so when that's a live risk.

THIRD, A DETECTION FAILURE IS NOT A MEASUREMENT ERROR. If the circle finder
misses a coin or finds a reflection, that's a different bug from measuring a
found coin badly. Unmatched detections and unfound expectations are counted and
reported separately, never folded into the error distribution.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pipeline.calibrate import CalibrationSettings, calibrate

# Sizes I can defend. The card is a published standard. The coins are the Royal
# Canadian Mint's specifications. Letter paper is the ANSI/ASME Y14.1 size.
# None of these came off a ruler in my kitchen.
KNOWN_OBJECTS: dict[str, dict[str, Any]] = {
    "loonie":  {"shape": "circle", "diameter_mm": 26.50, "source": "Royal Canadian Mint $1"},
    "toonie":  {"shape": "circle", "diameter_mm": 28.00, "source": "Royal Canadian Mint $2"},
    "quarter": {"shape": "circle", "diameter_mm": 23.88, "source": "Royal Canadian Mint 25c"},
    "dime":    {"shape": "circle", "diameter_mm": 18.03, "source": "Royal Canadian Mint 10c"},
    "nickel":  {"shape": "circle", "diameter_mm": 21.20, "source": "Royal Canadian Mint 5c"},
    "letter":  {"shape": "rect", "width_mm": 215.90, "height_mm": 279.40, "source": "ANSI A"},
    "card":    {"shape": "rect", "width_mm": 85.60, "height_mm": 53.98, "source": "ISO/IEC 7810 ID-1"},
}


@dataclass
class ObjectMeasurement:
    """One object, in one frame, measured against its known size."""

    image: str
    expected: str
    truth_mm: float
    measured_mm: float
    error_mm: float
    error_pct: float
    dimension: str          # "diameter" | "width" | "height"
    obliquity: float | None
    residual_px: float | None
    calibrated: bool

    @property
    def abs_error_mm(self) -> float:
        return abs(self.error_mm)


# ---------------------------------------------------------------------------
# finding the things that aren't the calibration target
# ---------------------------------------------------------------------------


def _grey(frame: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return cv2.GaussianBlur(g, (5, 5), 0)


def find_circular_contours(
    frame: np.ndarray,
    min_area_frac: float = 0.0002,
    max_area_frac: float = 0.25,
    min_roundness: float = 0.85,
    min_axis_ratio: float = 0.45,
    exclude: np.ndarray | None = None,
    n_levels: int = 48,
) -> list[np.ndarray]:
    """Closed contours round enough to be a coin, in pixel space.

    This used to threshold once with Otsu and it failed on the first real
    photograph I took. Otsu picks ONE threshold and assumes the image is
    bimodal. A tabletop isn't: that scene had grey carpet at 80, one loonie at
    136, another loonie at 187 and the card at 227, and Otsu landed on 146,
    straight between the two coins. One coin went to the background, the other
    merged with the card, and the detector reported zero.

    So sweep the threshold instead of guessing it. Anything that comes back
    round, and the right sort of size, at ANY level is a candidate. That's the
    stable-region idea behind MSER, minus the machinery: a real object stays
    the same shape across a range of thresholds, while noise appears at one
    level and vanishes at the next.

    Roundness here is area-based, not perimeter-based, and that's the second
    thing the real photograph corrected. I started with circularity,
    4*pi*A / P^2. Perimeter is exactly the quantity that noise inflates: a
    coin whose edge goes ragged where it blends into carpet scored 0.697 and
    was thrown away, while its AREA was 133,972 px^2 against an expected
    136,000. The shape was fine; the outline was fuzzy.

    So fit an ellipse and compare the contour's area to the ellipse's. Area is
    an integral over the whole region and barely notices a ragged boundary,
    which is the same argument that makes equivalent_diameter_mm use area
    instead of minEnclosingCircle. The axis-ratio bar separately rejects
    anything too foreshortened to be a coin viewed from a sane angle.

    `exclude` is the calibration target's outline. A coin detector that happily
    returns the card is measuring its own ruler.
    """
    h, w = frame.shape[:2]
    lo_area, hi_area = min_area_frac * h * w, max_area_frac * h * w
    grey = _grey(frame)

    found: list[tuple[float, float, float, np.ndarray]] = []  # cx, cy, area, contour
    for level in np.linspace(20, 240, n_levels):
        _, binary = cv2.threshold(grey, float(level), 255, cv2.THRESH_BINARY)
        for polarity in (binary, cv2.bitwise_not(binary)):
            contours, _ = cv2.findContours(
                polarity, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            for c in contours:
                area = abs(cv2.contourArea(c))
                if area < lo_area or area > hi_area:
                    continue
                if len(c) < 5:
                    continue  # fitEllipse needs five points
                (_, _), (d1, d2), _ = cv2.fitEllipse(c)
                major, minor = max(d1, d2), min(d1, d2)
                if major <= 0 or minor / major < min_axis_ratio:
                    continue
                ellipse_area = math.pi * major * minor / 4.0
                if ellipse_area <= 0 or area / ellipse_area < min_roundness:
                    continue
                pts = c.reshape(-1, 2).astype(np.float64)
                cx, cy = pts.mean(axis=0)
                if exclude is not None and cv2.pointPolygonTest(
                    exclude.astype(np.float32), (float(cx), float(cy)), False
                ) >= 0:
                    continue  # that's the calibration card
                found.append((float(cx), float(cy), area, pts))

    # One object shows up at many levels. Cluster by centre and keep the MEDIAN
    # area from each cluster rather than the biggest or the roundest. The
    # extremes are the levels where the threshold was starting to eat into the
    # object or bleed out of it, and the middle is the stable part.
    clusters: list[list[tuple[float, float, float, np.ndarray]]] = []
    for item in found:
        for cl in clusters:
            if math.hypot(item[0] - cl[0][0], item[1] - cl[0][1]) < math.sqrt(cl[0][2]) / 2:
                cl.append(item)
                break
        else:
            clusters.append([item])

    out: list[np.ndarray] = []
    for cl in clusters:
        if len(cl) < 3:
            continue  # appeared at one or two levels only: that's noise, not an object
        cl.sort(key=lambda t: t[2])
        out.append(cl[len(cl) // 2][3])
    return out


def equivalent_diameter_mm(contour_px: np.ndarray, calib) -> float:
    """Diameter of the circle with the same area, measured in the board plane.

    Deliberately not minEnclosingCircle: that is decided by the two most extreme
    boundary pixels, so one speck of glare on a coin's rim moves it. Area is an
    integral over the whole outline and barely notices the same speck.

    Mapping to millimetres BEFORE measuring is the point. A coin viewed at an
    angle is an ellipse in the image; the homography undoes exactly that, so in
    the board plane it is a circle again, provided it lies on the board plane,
    which is the assumption this whole module exists to put a number on.
    """
    pts_mm = calib.to_mm(contour_px)
    area_mm2 = abs(cv2.contourArea(pts_mm.astype(np.float32)))
    return 2.0 * math.sqrt(area_mm2 / math.pi)


# ---------------------------------------------------------------------------
# one frame
# ---------------------------------------------------------------------------


def validate_image(
    image_path: str | Path,
    expected: list[str],
    settings: CalibrationSettings,
) -> tuple[list[ObjectMeasurement], list[str]]:
    """Measure the expected objects in one image. Returns (measurements, notes)."""
    path = Path(image_path)
    frame = cv2.imread(str(path))
    notes: list[str] = []
    if frame is None:
        return [], [f"{path.name}: could not read the image"]

    calib = calibrate(frame, settings)
    if not calib.calibrated:
        return [], [f"{path.name}: uncalibrated, {calib.reason}"]
    if not calib.reliable:
        notes.append(
            f"{path.name}: calibration is UNRELIABLE "
            f"(obliquity {calib.obliquity:.2f}, residual {calib.residual_px}). "
            f"measurements included but flagged"
        )

    circles = [e for e in expected if KNOWN_OBJECTS[e]["shape"] == "circle"]
    found = find_circular_contours(frame, exclude=calib.corners_px)

    measured = sorted(
        ((equivalent_diameter_mm(c, calib), c) for c in found), key=lambda t: t[0]
    )
    truths = sorted(circles, key=lambda n: KNOWN_OBJECTS[n]["diameter_mm"])

    # Throw away detections that cannot be any expected object. This is not
    # cherry-picking: it rejects things outside half to twice the whole expected
    # range, which no amount of measurement error would produce. One of my real
    # shots found a round object on the floor behind the table and measured it at
    # 9.3 mm; paired by rank against a 26.50 mm loonie that is a -17 mm "error"
    # that is really a detection artifact on a different plane.
    #
    # The risk is worth stating: if the calibration were badly wrong, every real
    # object would fall outside this gate and get dropped, and the harness would
    # report nothing instead of reporting nonsense. That's the behaviour I want,
    # but "no measurements" should be read as a warning, not as a pass.
    if truths:
        lo = 0.5 * min(KNOWN_OBJECTS[n]["diameter_mm"] for n in truths)
        hi = 2.0 * max(KNOWN_OBJECTS[n]["diameter_mm"] for n in truths)
        rejected = [d for d, _ in measured if not (lo <= d <= hi)]
        measured = [(d, c) for d, c in measured if lo <= d <= hi]
        if rejected:
            notes.append(
                f"{path.name}: discarded {len(rejected)} detection(s) too far from any "
                f"expected size to be one ({', '.join(f'{d:.1f}mm' for d in rejected)}). "
                f"counted as detection failures, not measurement error"
            )

    if len(measured) != len(truths):
        notes.append(
            f"{path.name}: expected {len(truths)} object(s), matched {len(measured)}. "
            f"reporting NO measurements for this frame. Pairing a wrong number of "
            f"detections by rank would attribute one object's size to another and "
            f"call the difference measurement error."
        )
        return [], notes

    out: list[ObjectMeasurement] = []
    for name, (dia_mm, _c) in zip(truths, measured):
        truth = float(KNOWN_OBJECTS[name]["diameter_mm"])
        err = dia_mm - truth
        out.append(
            ObjectMeasurement(
                image=path.name,
                expected=name,
                truth_mm=truth,
                measured_mm=round(dia_mm, 4),
                error_mm=round(err, 4),
                error_pct=round(100.0 * err / truth, 4),
                dimension="diameter",
                obliquity=None if calib.obliquity is None else round(calib.obliquity, 4),
                residual_px=None if calib.residual_px is None else round(calib.residual_px, 4),
                calibrated=calib.reliable,
            )
        )
    return out, notes


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------


def rank_matching_is_risky(expected: list[str], spread_mm: float) -> list[str]:
    """Name the pairs close enough in true size that rank matching could swap
    them, given the error spread actually observed."""
    circles = sorted(
        (n for n in expected if KNOWN_OBJECTS[n]["shape"] == "circle"),
        key=lambda n: KNOWN_OBJECTS[n]["diameter_mm"],
    )
    risky = []
    for a, b in zip(circles, circles[1:]):
        gap = KNOWN_OBJECTS[b]["diameter_mm"] - KNOWN_OBJECTS[a]["diameter_mm"]
        if gap < 2.0 * spread_mm:
            risky.append(f"{a}/{b} differ by {gap:.2f} mm")
    return risky


def summarise(
    rows: list[ObjectMeasurement], working_distance_mm: float | None = None
) -> dict[str, Any]:
    """The distribution, and the coplanarity bias prediction if I can make one."""
    if not rows:
        return {"n": 0}
    err = np.array([r.error_mm for r in rows])
    pct = np.array([r.error_pct for r in rows])
    s: dict[str, Any] = {
        "n": len(rows),
        "bias_mm": float(err.mean()),          # signed: the systematic part
        "bias_pct": float(pct.mean()),
        "mae_mm": float(np.abs(err).mean()),
        "rms_mm": float(np.sqrt((err ** 2).mean())),
        "p95_abs_mm": float(np.percentile(np.abs(err), 95)),
        "max_abs_mm": float(np.abs(err).max()),
        "sd_mm": float(err.std(ddof=1)) if len(err) > 1 else 0.0,
    }
    if working_distance_mm:
        # A coin's top face sits about 1 mm proud of a card's. Apparent size
        # scales by D/(D-h), so the predicted over-read is roughly h/D.
        h = 1.0
        s["predicted_coplanarity_bias_pct"] = 100.0 * h / float(working_distance_mm)
        s["predicted_coplanarity_bias_mm"] = (
            s["predicted_coplanarity_bias_pct"] / 100.0
            * float(np.mean([r.truth_mm for r in rows]))
        )
    return s


def write_csv(rows: list[ObjectMeasurement], out_path: str | Path) -> Path:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()) if rows else ["image"])
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    return p


def main(argv: list[str] | None = None) -> int:
    import argparse

    from pipeline.config import load_config

    ap = argparse.ArgumentParser(
        prog="python -m eval.validate_mm",
        description="Measure objects of known size and report the error distribution.",
    )
    ap.add_argument("shots", help="an image, or a directory of images")
    ap.add_argument(
        "--objects", required=True,
        help="comma-separated names present in EVERY shot, e.g. loonie,toonie,quarter",
    )
    ap.add_argument("--out", default="data/exports/validation_mm.csv")
    ap.add_argument("--config", default=None)
    ap.add_argument(
        "--working-distance-mm", type=float, default=None,
        help="camera to table, roughly. Only used to predict the coplanarity bias.",
    )
    args = ap.parse_args(argv)

    expected = [s.strip() for s in args.objects.split(",") if s.strip()]
    unknown = [e for e in expected if e not in KNOWN_OBJECTS]
    if unknown:
        print(f"I don't have a published size for: {', '.join(unknown)}")
        print(f"Known: {', '.join(sorted(KNOWN_OBJECTS))}")
        return 2

    cfg = load_config(args.config)
    settings = CalibrationSettings.from_config(cfg)
    # The card IS the ruler here, so never let ArUco win the auto race.
    settings = CalibrationSettings(**{**settings.__dict__, "target": "card"})

    p = Path(args.shots)
    images = (
        sorted([q for q in p.iterdir() if q.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        if p.is_dir() else [p]
    )
    if not images:
        print(f"no images under {p}")
        return 2

    rows: list[ObjectMeasurement] = []
    notes: list[str] = []
    for img in images:
        r, n = validate_image(img, expected, settings)
        rows.extend(r)
        notes.extend(n)

    print(f"\n{len(images)} image(s), {len(rows)} measurement(s)\n")
    for n in notes:
        print(f"  note: {n}")
    if notes:
        print()

    if not rows:
        print("Nothing measured. Most likely the card wasn't found. Check that it's")
        print("fully in frame, not touching the edge, and against a contrasting surface.")
        return 1

    s = summarise(rows, args.working_distance_mm)
    print(f"{'object':<10}{'truth':>9}{'measured':>10}{'error':>9}{'error %':>9}")
    for name in sorted({r.expected for r in rows}):
        sub = [r for r in rows if r.expected == name]
        e = np.array([r.error_mm for r in sub])
        print(
            f"{name:<10}{sub[0].truth_mm:>9.2f}"
            f"{np.mean([r.measured_mm for r in sub]):>10.2f}"
            f"{e.mean():>+9.3f}{np.mean([r.error_pct for r in sub]):>+9.3f}"
            f"   (n={len(sub)})"
        )

    print(
        f"\noverall   bias {s['bias_mm']:+.3f} mm ({s['bias_pct']:+.3f}%)   "
        f"MAE {s['mae_mm']:.3f} mm   RMS {s['rms_mm']:.3f} mm   "
        f"p95 |err| {s['p95_abs_mm']:.3f} mm   worst {s['max_abs_mm']:.3f} mm"
    )

    if "predicted_coplanarity_bias_pct" in s:
        print(
            f"\ncoplanarity: a coin sits ~1 mm above the card, so at "
            f"{args.working_distance_mm:.0f} mm working distance I'd predict a "
            f"+{s['predicted_coplanarity_bias_pct']:.3f}% over-read "
            f"(+{s['predicted_coplanarity_bias_mm']:.3f} mm). "
            f"Measured bias is {s['bias_pct']:+.3f}%."
        )

    risky = rank_matching_is_risky(expected, s["sd_mm"])
    if risky:
        print(
            "\nrank matching warning: these are close enough in true size that a "
            "swap is possible at the observed error spread: " + "; ".join(risky)
        )

    out = write_csv(rows, args.out)
    print(f"\nwrote {len(rows)} rows to {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
