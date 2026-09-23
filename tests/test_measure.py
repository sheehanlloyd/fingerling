"""Measurement tests.

Every fixture is built from the answer: tests/fish.py lays out a fish whose fork
length is exactly 200 mm and whose body depth is exactly 50 mm, projects it into
pixels through a scale I chose, and the test checks the code gets those numbers
back. A test that only asserted "returns a MeasurementSet" would pass on code
that computed nonsense, which is the failure mode I care about.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.measure import MeasureSettings, measure_traits
from tests import fish as F

SET = MeasureSettings()


# ---------------------------------------------------------------------------
# lengths
# ---------------------------------------------------------------------------


def test_a_200mm_fish_measures_200mm():
    m = measure_traits(F.place(), F.scale_calibration(), SET)
    assert m.fork_length_mm is not None
    assert m.fork_length_mm.value == pytest.approx(F.FORK_LENGTH_MM, abs=0.01)


def test_the_same_fish_measures_200mm_at_a_different_pixel_scale():
    """Doubling the pixels per millimetre must not change the millimetres. This
    is the test that would catch a scale factor applied twice or not at all."""
    a = measure_traits(F.place(mm_per_px=0.5), F.scale_calibration(0.5), SET)
    b = measure_traits(F.place(mm_per_px=0.25), F.scale_calibration(0.25), SET)
    assert a.fork_length_mm.value == pytest.approx(200.0, abs=0.01)
    assert b.fork_length_mm.value == pytest.approx(200.0, abs=0.01)


def test_rotating_the_fish_does_not_change_any_length():
    flat = measure_traits(F.place(angle_deg=0), F.scale_calibration(), SET)
    tilted = measure_traits(F.place(angle_deg=37.0), F.scale_calibration(), SET)
    assert tilted.fork_length_mm.value == pytest.approx(flat.fork_length_mm.value, abs=0.01)
    assert tilted.body_depth_mm.value == pytest.approx(flat.body_depth_mm.value, abs=0.01)


def test_body_and_peduncle_depth_match_the_construction():
    m = measure_traits(F.place(), F.scale_calibration(), SET)
    assert m.body_depth_mm.value == pytest.approx(F.BODY_DEPTH_MM, abs=0.01)
    assert m.peduncle_depth_mm.value == pytest.approx(F.PEDUNCLE_DEPTH_MM, abs=0.01)


def test_total_length_is_longer_than_fork_length():
    """The tail tip is behind the fork, so this ordering is a fact about fish and
    a cheap check that the two traits aren't reading the same landmark pair."""
    m = measure_traits(F.place(), F.scale_calibration(), SET)
    assert m.total_length_mm.value == pytest.approx(F.TOTAL_LENGTH_MM, abs=0.01)
    assert m.total_length_mm.value > m.fork_length_mm.value


# ---------------------------------------------------------------------------
# ratios, and the property that makes them worth having
# ---------------------------------------------------------------------------


def test_depth_ratio_matches_the_construction():
    m = measure_traits(F.place(), F.scale_calibration(), SET)
    assert m.depth_ratio.value == pytest.approx(F.DEPTH_RATIO, abs=1e-6)


def test_depth_ratio_is_identical_with_no_calibration_at_all():
    """The point of a dimensionless trait: it survives a frame where the
    calibration target was missing entirely."""
    with_cal = measure_traits(F.place(), F.scale_calibration(), SET)
    without = measure_traits(F.place(), F.uncalibrated(), SET)
    assert without.fork_length_mm is None
    assert without.depth_ratio.value == pytest.approx(with_cal.depth_ratio.value, abs=1e-9)


def test_uncalibrated_still_reports_pixels_and_says_why_millimetres_are_missing():
    m = measure_traits(F.place(), F.uncalibrated(), SET)
    assert m.calibrated is False
    assert m.fork_length_px is not None and m.fork_length_px.value > 0
    assert "not calibrated" in m.unavailable["fork_length_mm"]


# ---------------------------------------------------------------------------
# curvature, the deformity proxy
# ---------------------------------------------------------------------------


def test_a_straight_fish_scores_near_zero_curvature():
    m = measure_traits(F.place(bend_mm=0.0), F.scale_calibration(), SET)
    assert m.curvature_index.value < 0.005


def test_a_bent_fish_scores_well_above_a_straight_one():
    straight = measure_traits(F.place(bend_mm=0.0), F.scale_calibration(), SET)
    bent = measure_traits(F.place(bend_mm=20.0), F.scale_calibration(), SET)
    assert bent.curvature_index.value > straight.curvature_index.value + 0.05


def test_curvature_scales_with_how_bent_the_fish_is():
    """Not just "bent is bigger". Twice the bow should read about twice the
    index, because the index is a linear ratio of deviation to length."""
    a = measure_traits(F.place(bend_mm=10.0), F.scale_calibration(), SET)
    b = measure_traits(F.place(bend_mm=20.0), F.scale_calibration(), SET)
    assert b.curvature_index.value == pytest.approx(2 * a.curvature_index.value, rel=0.1)


def test_a_20mm_bow_on_a_200mm_fish_reads_about_0_1():
    """The index is deviation over fork length, so I can state the answer in
    advance. The midline point nearest the middle of the fish carries the full
    20 mm bow, and 20/200 is 0.1."""
    m = measure_traits(F.place(bend_mm=20.0), F.scale_calibration(), SET)
    assert m.curvature_index.value == pytest.approx(0.1, rel=0.15)


def test_curvature_does_not_care_which_way_the_fish_is_lying():
    a = measure_traits(F.place(bend_mm=15.0, angle_deg=0.0), F.scale_calibration(), SET)
    b = measure_traits(F.place(bend_mm=15.0, angle_deg=145.0), F.scale_calibration(), SET)
    assert b.curvature_index.value == pytest.approx(a.curvature_index.value, rel=1e-6)


# ---------------------------------------------------------------------------
# uncertainty
# ---------------------------------------------------------------------------


def test_lower_confidence_widens_the_error_bar():
    good = measure_traits(F.place(confidence=0.95), F.scale_calibration(), SET)
    poor = measure_traits(F.place(confidence=0.30), F.scale_calibration(), SET)
    assert poor.fork_length_mm.sigma > good.fork_length_mm.sigma * 2


def test_a_smaller_calibration_target_widens_the_error_bar_on_a_length():
    """Corner error divided by target span is the relative scale uncertainty, so
    extrapolating a 200 mm fish off a small target has to cost something. If this
    test fails the calibration term isn't reaching the result at all."""
    big = measure_traits(F.place(), F.scale_calibration(target_span_px=800.0), SET)
    small = measure_traits(F.place(), F.scale_calibration(target_span_px=60.0), SET)
    assert small.fork_length_mm.sigma > big.fork_length_mm.sigma


def test_calibration_error_does_not_touch_the_depth_ratio():
    """Scale cancels in a ratio. That's the entire argument for preferring
    ratios, so it gets a test rather than a comment."""
    big = measure_traits(F.place(), F.scale_calibration(target_span_px=800.0), SET)
    small = measure_traits(F.place(), F.scale_calibration(target_span_px=60.0), SET)
    assert small.depth_ratio.sigma == pytest.approx(big.depth_ratio.sigma, rel=1e-9)


def test_the_error_bar_on_a_200mm_length_is_not_absurd():
    """A sanity band, not a claim. At confidence 0.9 the landmark sigma is about
    1.3 px, which at 0.5 mm/px is well under a millimetre per endpoint, so the
    length uncertainty belongs in the sub-millimetre range. If this ever reports
    centimetres something is multiplying by the wrong scale."""
    m = measure_traits(F.place(confidence=0.9), F.scale_calibration(), SET)
    assert 0.0 < m.fork_length_mm.sigma < 2.0


# ---------------------------------------------------------------------------
# condition factor
# ---------------------------------------------------------------------------


def test_condition_factor_uses_centimetres_so_a_healthy_fish_scores_near_one():
    """Fulton's K = 100 W / L^3 with W in grams and L in cm. 100 g at 20 cm is
    100 * 100 / 8000 = 1.25. In millimetres it would come out at 1.25e-6, which
    is the mistake this test exists to catch."""
    m = measure_traits(F.place(), F.scale_calibration(), SET, weight_g=100.0)
    assert m.condition_factor.value == pytest.approx(1.25, rel=0.01)


def test_no_weight_means_no_condition_factor_and_a_reason():
    m = measure_traits(F.place(), F.scale_calibration(), SET)
    assert m.condition_factor is None
    assert m.unavailable["condition_factor"] == "no weight entered"


# ---------------------------------------------------------------------------
# graceful degradation
# ---------------------------------------------------------------------------


def test_a_missing_landmark_nulls_only_the_traits_that_need_it():
    """Drop the fork point and fork length has to go away, not silently become
    total length. Body depth doesn't depend on it and must survive."""
    m = measure_traits(F.place(drop=("caudal_fork",)), F.scale_calibration(), SET)
    assert m.fork_length_mm is None
    assert "caudal_fork" in m.unavailable["fork_length_mm"]
    assert m.body_depth_mm.value == pytest.approx(F.BODY_DEPTH_MM, abs=0.01)
    assert m.total_length_mm is not None


def test_curvature_needs_three_midline_points_and_says_so_when_it_cannot():
    m = measure_traits(
        F.place(drop=("dorsal_origin", "ventral_margin", "peduncle_dorsal",
                      "peduncle_ventral")),
        F.scale_calibration(),
        SET,
    )
    assert m.curvature_index is None
    assert "curvature_index" in m.unavailable


def test_no_detection_produces_no_traits_and_does_not_raise():
    from pipeline.landmarks import Landmarks

    m = measure_traits(Landmarks.empty(), F.scale_calibration(), SET)
    assert m.fork_length_mm is None
    assert m.depth_ratio is None
    assert m.unavailable["fork_length_mm"] == "no detection"
