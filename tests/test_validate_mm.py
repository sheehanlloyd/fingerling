"""Tests for the millimetre validation harness.

These prove the GEOMETRY, not the accuracy. Every scene here is rendered through
a homography I chose, so the answer is known exactly and any disagreement is my
code's fault. What they cannot tell me is how the thing behaves on a photograph:
no lens, no sensor noise, no coin thickness, no lighting. That's what the real
shots are for, and no amount of synthetic testing substitutes for them.

The tolerance below is 0.25 mm and it is deliberately tight. The boundary bias
this harness found in its own segmentation (thresholding cut ~0.41 mm inside
every object) would fail these tests. That's the point of picking a number that
a broken version can't sneak under.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval.validate_mm import (
    KNOWN_OBJECTS,
    equivalent_diameter_mm,
    find_circular_contours,
    rank_matching_is_risky,
    summarise,
    validate_image,
)
from pipeline.calibrate import CalibrationSettings, calibrate
from tests.synthetic import card_and_coins_scene, foreshorten, quad, rotate

# Board-plane layout: (centre_x_mm, centre_y_mm, true_diameter_mm), all below the
# card so they don't interfere with detecting it. Spaced so they never touch,
# two coins sharing a contour is a detection bug, not a measurement one.
COINS = [(15.0, 95.0, 26.50), (52.0, 95.0, 28.00), (88.0, 95.0, 23.88)]
TRUE_MM = sorted(c[2] for c in COINS)
CARD_QUAD = quad(700, 300, 420, 265)

TOLERANCE_MM = 0.25


def measure(scene_quad, max_obliquity: float = 99.0):
    """Render, calibrate, find the coins, return their diameters smallest-first."""
    frame = card_and_coins_scene(scene_quad, COINS)
    calib = calibrate(
        frame, CalibrationSettings(target="card", max_obliquity=max_obliquity)
    )
    found = find_circular_contours(frame, exclude=calib.corners_px)
    return frame, calib, sorted(equivalent_diameter_mm(c, calib) for c in found)


def test_a_coin_of_known_size_measures_that_size_square_on():
    """The base case. A 26.50 mm loonie has to come back 26.50 mm."""
    _, calib, measured = measure(CARD_QUAD)
    assert calib.calibrated and calib.reliable
    assert len(measured) == 3, "expected three coins, detection is the wrong test"
    for truth, got in zip(TRUE_MM, measured):
        assert abs(got - truth) < TOLERANCE_MM, f"{truth} mm measured as {got:.3f} mm"


def test_in_plane_rotation_is_a_rigid_motion_and_must_change_nothing():
    """Rotating the whole scene in the image plane adds no perspective. If these
    numbers move, something is reading an axis-aligned assumption it shouldn't."""
    _, _, flat = measure(CARD_QUAD)
    _, calib, turned = measure(rotate(CARD_QUAD, 30.0))
    assert calib.obliquity == pytest.approx(1.0, abs=0.05), "rotation is not tilt"
    for a, b in zip(flat, turned):
        assert abs(a - b) < 0.05, f"rotation moved a measurement: {a:.3f} -> {b:.3f}"


def test_perspective_tilt_is_undone_by_the_homography():
    """A tilted board still measures right. This is the entire argument for
    solving a homography rather than dividing by a pixels-per-mm constant. A
    scale factor cannot undo foreshortening, and a homography can."""
    _, calib, measured = measure(foreshorten(CARD_QUAD, 0.85))
    assert calib.obliquity > 1.2, "scene isn't actually tilted"
    assert len(measured) == 3
    for truth, got in zip(TRUE_MM, measured):
        assert abs(got - truth) < TOLERANCE_MM, f"{truth} mm measured as {got:.3f} mm"


def test_a_naive_pixels_per_mm_would_fail_the_tilted_case():
    """Guards the claim above rather than asserting it in prose.

    Take the card's mean pixel width, divide it into 85.60 mm, and apply that one
    scale factor to a tilted coin the way a naive implementation would. It should
    be visibly wrong, because otherwise the tilted test proves nothing: a
    constant would have passed it too.
    """
    frame, calib, _ = measure(foreshorten(CARD_QUAD, 0.85))
    q = calib.corners_px
    width_px = (np.linalg.norm(q[0] - q[1]) + np.linalg.norm(q[3] - q[2])) / 2
    naive_mm_per_px = 85.60 / width_px

    found = find_circular_contours(frame, exclude=q)
    areas_px = sorted(_area(c) for c in found)
    naive = [2.0 * np.sqrt(a / np.pi) * naive_mm_per_px for a in areas_px]
    worst = max(abs(n - t) for n, t in zip(naive, TRUE_MM))
    assert worst > TOLERANCE_MM, (
        f"a constant scale factor was accurate to {worst:.3f} mm on this scene, so "
        f"the tilted test doesn't demonstrate the homography is doing anything"
    )


def _area(contour: np.ndarray) -> float:
    import cv2

    return abs(cv2.contourArea(contour.astype(np.float32)))


def test_objects_at_different_brightnesses_are_all_found():
    """The failure a real photograph produced and no synthetic scene had.

    The detector used to threshold once with Otsu. Otsu picks a single level and
    assumes the image is bimodal; my first real shot had carpet at 80, one coin
    at 136, another at 187 and the card at 227, and Otsu landed on 146, exactly
    between the two coins. One went to the background, the other merged with the
    card, and it reported zero objects.

    This scene reproduces that: three coins at three brightnesses straddling
    whatever single threshold you'd pick.
    """
    import cv2

    frame = card_and_coins_scene(CARD_QUAD, COINS, coin_grey=[90, 150, 205])
    calib = calibrate(frame, CalibrationSettings(target="card", max_obliquity=99))

    grey = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    otsu, _ = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    assert 90 < otsu < 205, "scene doesn't actually straddle the Otsu level"

    found = find_circular_contours(frame, exclude=calib.corners_px)
    assert len(found) == 3, f"a single threshold can't see all three; found {len(found)}"
    for truth, got in zip(TRUE_MM, sorted(equivalent_diameter_mm(c, calib) for c in found)):
        assert abs(got - truth) < TOLERANCE_MM


def test_a_ragged_outline_does_not_lose_the_object():
    """Second thing the photograph corrected: roundness has to be area-based.

    Perimeter-based circularity is exactly the wrong statistic, because noise
    inflates perimeter fast. A real coin whose edge blended into carpet scored
    0.697 circularity and was discarded, while its area was 133,972 px^2
    against an expected 136,000. The shape was fine; the outline was fuzzy.
    """
    import cv2

    frame = card_and_coins_scene(CARD_QUAD, COINS)
    calib = calibrate(frame, CalibrationSettings(target="card", max_obliquity=99))

    rng = np.random.default_rng(0)
    noisy = frame.copy()
    noisy = np.clip(noisy.astype(np.int16) + rng.normal(0, 14, noisy.shape), 0, 255).astype(np.uint8)

    found = find_circular_contours(noisy, exclude=calib.corners_px)
    assert len(found) == 3, f"lost an object to a fuzzy edge; found {len(found)}"


def test_the_calibration_card_is_never_measured_as_a_coin():
    """A measurement system that measures its own ruler and calls it an object
    will report 85.60 mm with total confidence, forever."""
    frame = card_and_coins_scene(CARD_QUAD, COINS)
    calib = calibrate(frame, CalibrationSettings(target="card"))
    found = find_circular_contours(frame, exclude=calib.corners_px)
    diameters = [equivalent_diameter_mm(c, calib) for c in found]
    assert all(d < 40.0 for d in diameters), f"something card-sized got measured: {diameters}"


def test_no_card_means_no_measurement_rather_than_a_wrong_one(tmp_path):
    """Uncalibrated has to mean silence, not pixels wearing a millimetre label."""
    import cv2

    from tests.synthetic import blank_scene

    p = tmp_path / "nocard.png"
    cv2.imwrite(str(p), blank_scene())
    rows, notes = validate_image(p, ["loonie"], CalibrationSettings(target="card"))
    assert rows == []
    assert any("uncalibrated" in n for n in notes)


def test_a_missing_coin_produces_no_measurements_rather_than_a_wrong_pairing(tmp_path):
    """Expecting three and finding two must not produce two rows.

    Matching is by rank, so a missing object shifts every remaining one onto the
    wrong truth and the differences get reported as measurement error. On a real
    shot the reverse happened: a round object on the floor behind the table was
    detected, rank-paired against a 26.50 mm loonie, and came out as a -17 mm
    "error" that was really a detection on a different plane.
    """
    import cv2

    p = tmp_path / "two.png"
    cv2.imwrite(str(p), card_and_coins_scene(CARD_QUAD, COINS[:2]))
    rows, notes = validate_image(
        p, ["loonie", "toonie", "quarter"], CalibrationSettings(target="card")
    )
    assert rows == [], "a wrong number of detections must not be paired by rank"
    assert any("reporting NO measurements" in n for n in notes)


def test_a_detection_that_cannot_be_any_expected_object_is_discarded(tmp_path):
    """The floor object again, as a test. It's dropped as a detection failure
    and named in the notes, never folded into the error distribution."""
    import cv2

    scene = card_and_coins_scene(CARD_QUAD, COINS + [(120.0, 20.0, 6.0)])
    p = tmp_path / "extra.png"
    cv2.imwrite(str(p), scene)
    rows, notes = validate_image(
        p, ["loonie", "toonie", "quarter"], CalibrationSettings(target="card")
    )
    assert any("too far from any expected size" in n for n in notes)
    assert all(r.measured_mm > 15.0 for r in rows)


def test_rank_matching_warns_when_two_objects_are_too_close_in_size():
    """Loonie and toonie differ by 1.50 mm. At a 1 mm error spread, rank matching
    could swap them and the report has to say so instead of looking confident."""
    risky = rank_matching_is_risky(["loonie", "toonie", "quarter"], spread_mm=1.0)
    assert any("loonie/toonie" in r for r in risky)
    assert rank_matching_is_risky(["loonie", "toonie"], spread_mm=0.05) == []


def test_the_summary_separates_bias_from_scatter():
    """Bias is the systematic part and scatter is the random part, and they need
    different fixes: a bias survives averaging and scatter doesn't. A summary
    that reported only MAE would hide which one you have."""
    from eval.validate_mm import ObjectMeasurement

    rows = [
        ObjectMeasurement("a.jpg", "loonie", 26.5, 26.9, +0.4, 1.5, "diameter", 1.0, 0.5, True),
        ObjectMeasurement("b.jpg", "loonie", 26.5, 26.9, +0.4, 1.5, "diameter", 1.0, 0.5, True),
    ]
    s = summarise(rows)
    assert s["bias_mm"] == pytest.approx(0.4, abs=1e-6)
    assert s["sd_mm"] == pytest.approx(0.0, abs=1e-9)


def test_every_known_object_has_a_citable_source():
    """Rule: no number in the truth table came off a ruler in my kitchen. If one
    ever does, it needs saying out loud rather than sitting in a dict looking
    like a standard."""
    for name, spec in KNOWN_OBJECTS.items():
        assert spec.get("source"), f"{name} has no stated source"
        assert spec["shape"] in {"circle", "rect"}
