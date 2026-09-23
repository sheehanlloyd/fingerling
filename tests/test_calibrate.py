"""Calibration tests.

Every test here works the same way, and it's worth stating once: I pick a pair of
points in the board plane whose separation in millimetres I chose myself, project
them into the image with a homography I built, and then ask the calibration code
what that separation is. The answer has to come back as the number I started with.

That's the only kind of assertion I trust for this module. Checking that
`calibrate()` returns a Calibration object would pass on code that computed
nonsense.

Tolerances: the scenes are rendered and re-detected through pixel quantisation, so
the errors below aren't zero. Where I use a loose tolerance it's because the
target is small relative to the distance being measured and corner error gets
amplified — the extrapolation factor is stated in the test.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from pipeline.calibrate import (
    CalibrationSettings,
    CalibrationUnavailable,
    calibrate,
    plane_metrics,
)
from tests import synthetic as syn

MARKER_MM = 40.0

# A 200 mm span lying in the board plane, off to the side of the target — roughly
# where a fish would sit next to the calibration object. Measured from a 40 mm
# marker, that's a 5x extrapolation, so corner error gets multiplied by 5.
ARUCO_PROBE = np.array([[-80.0, 60.0], [120.0, 60.0]])
ARUCO_PROBE_MM = 200.0
ARUCO_PROBE_VERTICAL = np.array([[20.0, -40.0], [20.0, 160.0]])

CARD_PROBE = np.array([[-60.0, 90.0], [140.0, 90.0]])
CARD_PROBE_MM = 200.0


def aruco_settings(**kw) -> CalibrationSettings:
    base = dict(target="aruco", marker_length_mm=MARKER_MM)
    base.update(kw)
    return CalibrationSettings(**base)


def card_settings(**kw) -> CalibrationSettings:
    base = dict(target="card")
    base.update(kw)
    return CalibrationSettings(**base)


def measure(calib, probe_mm_pts, H_gt) -> float:
    """Project two known board points into pixels, then ask the calibration to
    turn them back into a millimetre distance."""
    px = syn.project(H_gt, probe_mm_pts)
    return calib.distance_mm(px[0], px[1])


# --------------------------------------------------------------------------
# the Jacobian math, checked against a case I can do on paper
# --------------------------------------------------------------------------


def test_plane_metrics_on_a_pure_scaling_is_exact():
    """A homography that's just "multiply pixels by 0.25 mm/px" has no tilt and
    no perspective, so obliquity must be exactly 1.0 and the scale exactly 0.25.
    If the Jacobian derivation is wrong this is where it shows."""
    H = np.array([[0.25, 0, 10.0], [0, 0.25, -3.0], [0, 0, 1.0]])
    obliquity, mm_per_px = plane_metrics(H, syn.quad(500, 400, 300, 200))
    assert obliquity == pytest.approx(1.0, abs=1e-9)
    assert mm_per_px == pytest.approx(0.25, abs=1e-9)


def test_plane_metrics_sees_anisotropic_scaling():
    """Different scale on each axis is a tilted plane. 0.5 mm/px one way and
    0.1 the other is an anisotropy of exactly 5."""
    H = np.array([[0.5, 0, 0.0], [0, 0.1, 0.0], [0, 0, 1.0]])
    obliquity, mm_per_px = plane_metrics(H, syn.quad(500, 400, 300, 200))
    assert obliquity == pytest.approx(5.0, abs=1e-9)
    # geometric mean of the two scales
    assert mm_per_px == pytest.approx(np.sqrt(0.5 * 0.1), abs=1e-9)


# --------------------------------------------------------------------------
# ArUco
# --------------------------------------------------------------------------


def test_aruco_square_on_measures_a_known_200mm_span():
    q = syn.quad(700, 450, 360, 360)
    frame, H_gt = syn.aruco_scene(q, MARKER_MM)
    calib = calibrate(frame, aruco_settings())

    assert calib.calibrated and calib.target == "aruco"
    assert measure(calib, ARUCO_PROBE, H_gt) == pytest.approx(ARUCO_PROBE_MM, abs=0.5)


def test_aruco_square_on_measures_the_same_in_both_directions():
    """A scale error on one axis only would pass the horizontal test and fail
    here."""
    q = syn.quad(700, 450, 360, 360)
    frame, H_gt = syn.aruco_scene(q, MARKER_MM)
    calib = calibrate(frame, aruco_settings())

    assert measure(calib, ARUCO_PROBE_VERTICAL, H_gt) == pytest.approx(200.0, abs=0.5)


def test_aruco_rotated_30_degrees_measures_the_same_span():
    """In-plane rotation is a rigid motion. The answer must not move at all."""
    q = syn.rotate(syn.quad(700, 450, 360, 360), 30.0)
    frame, H_gt = syn.aruco_scene(q, MARKER_MM)
    calib = calibrate(frame, aruco_settings())

    assert calib.calibrated
    assert measure(calib, ARUCO_PROBE, H_gt) == pytest.approx(ARUCO_PROBE_MM, abs=0.6)


def test_aruco_rotation_does_not_raise_obliquity():
    """The whole point of the obliquity metric is that it separates rotation
    (harmless) from tilt (not harmless). A rotated marker must still score ~1."""
    flat = syn.quad(700, 450, 360, 360)
    frame_flat, _ = syn.aruco_scene(flat, MARKER_MM)
    frame_rot, _ = syn.aruco_scene(syn.rotate(flat, 30.0), MARKER_MM)

    o_flat = calibrate(frame_flat, aruco_settings()).obliquity
    o_rot = calibrate(frame_rot, aruco_settings()).obliquity
    assert o_flat == pytest.approx(1.0, abs=0.02)
    assert o_rot == pytest.approx(1.0, abs=0.02)


def test_aruco_under_perspective_skew_still_measures_the_span():
    """Moderate tilt. The homography has real work to do here — an affine-only
    solve would be visibly wrong."""
    q = syn.foreshorten(syn.quad(700, 450, 420, 420), top_scale=0.72)
    frame, H_gt = syn.aruco_scene(q, MARKER_MM)
    calib = calibrate(frame, aruco_settings())

    assert calib.calibrated
    assert calib.obliquity > 1.05, "a tilted view should not score as flat"
    assert measure(calib, ARUCO_PROBE, H_gt) == pytest.approx(ARUCO_PROBE_MM, abs=3.0)


def test_marker_length_scales_every_measurement_linearly():
    """Tell the code the marker is twice as big and every millimetre figure must
    exactly double. This is the sharpest check I have that the physical scale
    enters in one place and enters correctly — a hardcoded constant or a squared
    scale factor anywhere in the chain fails this."""
    q = syn.quad(700, 450, 360, 360)
    frame, H_gt = syn.aruco_scene(q, MARKER_MM)

    px = syn.project(H_gt, ARUCO_PROBE)
    d_normal = calibrate(frame, aruco_settings()).distance_mm(px[0], px[1])
    d_double = calibrate(
        frame, aruco_settings(marker_length_mm=2 * MARKER_MM)
    ).distance_mm(px[0], px[1])

    assert d_double / d_normal == pytest.approx(2.0, rel=1e-9)


def test_aruco_residual_is_reported_but_marked_not_meaningful():
    """Four corners, eight unknowns: the residual is zero by construction. The
    flag has to say so, or someone downstream will read that zero as quality."""
    q = syn.foreshorten(syn.quad(700, 450, 420, 420), top_scale=0.5)
    frame, _ = syn.aruco_scene(q, MARKER_MM)
    calib = calibrate(frame, aruco_settings())

    assert calib.calibrated
    assert calib.residual_meaningful is False
    assert calib.residual_px < 1e-6, (
        "a 4-point homography reprojects exactly; if this is nonzero the solve "
        "or the inverse is wrong"
    )


def test_null_marker_length_refuses_to_calibrate():
    """config.yaml ships with marker_length_mm null because it depends on my
    phone's PPI. The code must not invent a scale to fill the gap."""
    q = syn.quad(700, 450, 360, 360)
    frame, _ = syn.aruco_scene(q, MARKER_MM)
    calib = calibrate(frame, aruco_settings(marker_length_mm=None))

    assert calib.calibrated is False
    assert "marker_length_mm" in calib.reason


# --------------------------------------------------------------------------
# credit card
# --------------------------------------------------------------------------


def test_card_square_on_measures_a_known_200mm_span():
    q = syn.quad(700, 450, 640, 640 * 53.98 / 85.60)
    frame, H_gt = syn.card_scene(q)
    calib = calibrate(frame, card_settings())

    assert calib.calibrated and calib.target == "card"
    assert measure(calib, CARD_PROBE, H_gt) == pytest.approx(CARD_PROBE_MM, abs=1.0)


def test_card_lying_portrait_measures_the_same_span():
    """The board frame is assigned by deciding which image edge is the card's
    long side. Get that backwards and every distance is off by 85.60/53.98 =
    1.586, which this catches immediately."""
    q = syn.rotate(syn.quad(700, 450, 640, 640 * 53.98 / 85.60), 90.0)
    frame, H_gt = syn.card_scene(q)
    calib = calibrate(frame, card_settings())

    assert calib.calibrated
    assert measure(calib, CARD_PROBE, H_gt) == pytest.approx(CARD_PROBE_MM, abs=1.5)


def test_card_rotated_30_degrees_measures_the_same_span():
    q = syn.rotate(syn.quad(700, 450, 620, 620 * 53.98 / 85.60), 30.0)
    frame, H_gt = syn.card_scene(q)
    calib = calibrate(frame, card_settings())

    assert calib.calibrated
    assert measure(calib, CARD_PROBE, H_gt) == pytest.approx(CARD_PROBE_MM, abs=1.5)


def test_card_under_perspective_skew_still_measures_the_span():
    q = syn.foreshorten(syn.quad(700, 450, 700, 700 * 53.98 / 85.60), top_scale=0.72)
    frame, H_gt = syn.card_scene(q)
    calib = calibrate(frame, card_settings())

    assert calib.calibrated
    assert calib.obliquity > 1.05
    assert measure(calib, CARD_PROBE, H_gt) == pytest.approx(CARD_PROBE_MM, abs=4.0)


def test_flat_card_outline_residual_is_small():
    """The card path gets a genuine over-determined residual from its outline.
    On a clean straight-edged card it should be well under a pixel."""
    q = syn.quad(700, 450, 640, 640 * 53.98 / 85.60)
    calib = calibrate(syn.card_scene(q)[0], card_settings())

    assert calib.residual_meaningful is True
    assert calib.residual_px < 1.0
    assert calib.reliable is True


def test_bowed_card_produces_a_large_residual_and_is_rejected():
    """A card that isn't flat — or a lens that bends straight lines — still gives
    four corners that solve a homography perfectly well. The corner-only residual
    cannot see this. The outline residual can, and that's the reason it exists.

    The numbers, measured on this scene at the configured 2.0 px threshold:

        bow 0.00 mm -> 0.88 px, accepted
        bow 1.00 mm -> 1.77 px, accepted
        bow 1.50 mm -> 2.34 px, REJECTED
        bow 3.00 mm -> 4.30 px, REJECTED

    So the detection limit sits somewhere around 1.2 mm of bow. That's the real
    sensitivity of this check and it's worth writing down rather than asserting
    some ratio — a card bowed by less than about a millimetre goes through, and
    its measurements are wrong by however much that costs.
    """
    q = syn.quad(700, 450, 640, 640 * 53.98 / 85.60)
    flat = calibrate(syn.card_scene(q)[0], card_settings())
    bowed = calibrate(syn.card_scene(q, bow_mm=1.5)[0], card_settings())

    assert bowed.calibrated is True, "it still finds a quad — that's the problem"
    assert flat.reliable is True
    assert bowed.residual_px > flat.max_residual_px
    assert bowed.reliable is False

    # Monotonic in the bow, so the number means something rather than just
    # happening to clear a threshold on one scene.
    series = [
        calibrate(syn.card_scene(q, bow_mm=b)[0], card_settings()).residual_px
        for b in (0.0, 0.5, 1.0, 1.5, 2.0)
    ]
    assert all(b > a for a, b in zip(series, series[1:])), series


# --------------------------------------------------------------------------
# bad and missing targets
# --------------------------------------------------------------------------


@pytest.mark.parametrize("scene", ["aruco", "card"])
def test_near_edge_on_view_is_flagged_unreliable(scene):
    """A steeply tilted target may still be detected and may still produce a
    homography. It must not be trusted. This is the failure mode that produces a
    confident wrong measurement, which is worse than no measurement."""
    q = syn.foreshorten(syn.quad(700, 450, 700, 700), top_scale=0.16, height_scale=0.55)
    if scene == "aruco":
        frame, _ = syn.aruco_scene(q, MARKER_MM)
        settings = aruco_settings()
    else:
        frame, _ = syn.card_scene(q)
        settings = card_settings()

    calib = calibrate(frame, settings)
    if not calib.calibrated:
        return  # detector gave up on it, which is also a correct outcome
    assert calib.obliquity > settings.max_obliquity
    assert calib.reliable is False


def test_tilt_amplifies_corner_noise_and_that_is_the_point_of_obliquity():
    """Why a steeply tilted target is dangerous, demonstrated rather than asserted.

    On perfectly rendered synthetic frames an edge-on marker actually measures
    fine — the homography is exact, so exact corners give exact millimetres. That
    is not what happens with a real camera. Real corner estimates are off by a
    fraction of a pixel, and a tilted homography multiplies that error far more
    than a flat one does.

    So here I skip the renderer, take exact corner positions, jitter them by half
    a pixel, and see what that does to a 200 mm measurement. Flat view: the error
    stays small. Edge-on: it grows several times over. Nothing in the reprojection
    residual sees any of this, which is exactly why `reliable` gates on obliquity.

    Both quads start from the same 700 px square so the comparison is about tilt
    and not about one target simply being bigger in frame. The probe runs across
    the far side of the target, where foreshortening is worst — measured
    amplification there is around 7x, and it's about 3x on the near side. I assert
    the weaker bound so the test isn't pinned to one random seed.
    """
    board = np.array(
        [[0, 0], [MARKER_MM, 0], [MARKER_MM, MARKER_MM], [0, MARKER_MM]],
        dtype=np.float64,
    )
    far_side_probe = np.array([[-80.0, -30.0], [120.0, -30.0]])

    def mean_error_under_noise(image_quad: np.ndarray) -> float:
        rng = np.random.default_rng(1)
        H_gt = syn.board_to_image(board, image_quad)
        probe_px = syn.project(H_gt, far_side_probe)
        errors = []
        for _ in range(400):
            noisy = image_quad + rng.normal(0.0, 0.5, image_quad.shape)
            H, _ = cv2.findHomography(noisy, board, method=0)
            p = cv2.perspectiveTransform(probe_px.reshape(-1, 1, 2), H).reshape(-1, 2)
            errors.append(abs(float(np.hypot(*(p[1] - p[0]))) - ARUCO_PROBE_MM))
        return float(np.mean(errors))

    base = syn.quad(700, 450, 700, 700)
    flat_quad = base
    tilted_quad = syn.foreshorten(base, 0.16, 0.55)

    flat_err = mean_error_under_noise(flat_quad)
    tilted_err = mean_error_under_noise(tilted_quad)

    assert tilted_err > 3 * flat_err, (
        f"expected tilt to amplify corner noise: flat {flat_err:.2f} mm, "
        f"tilted {tilted_err:.2f} mm"
    )
    # And the metric that's supposed to notice, notices.
    assert plane_metrics(np.linalg.inv(syn.board_to_image(board, flat_quad)),
                         flat_quad)[0] == pytest.approx(1.0, abs=0.01)
    assert plane_metrics(np.linalg.inv(syn.board_to_image(board, tilted_quad)),
                         tilted_quad)[0] > 5.0


def test_target_absent_returns_uncalibrated_and_does_not_raise():
    calib = calibrate(syn.blank_scene(), CalibrationSettings(target="auto",
                                                             marker_length_mm=MARKER_MM))
    assert calib.calibrated is False
    assert calib.H is None
    assert calib.reason


def test_uncalibrated_refuses_to_produce_millimetres():
    """Falling back to pixels silently would put pixel values in a field labelled
    millimetres. Better to raise and let the caller record 'uncalibrated'."""
    calib = calibrate(syn.blank_scene(), CalibrationSettings(target="auto",
                                                             marker_length_mm=MARKER_MM))
    with pytest.raises(CalibrationUnavailable):
        calib.to_mm(np.array([[10.0, 10.0]]))


def test_auto_mode_falls_back_from_marker_to_card():
    q = syn.quad(700, 450, 640, 640 * 53.98 / 85.60)
    frame, H_gt = syn.card_scene(q)
    calib = calibrate(
        frame, CalibrationSettings(target="auto", marker_length_mm=MARKER_MM)
    )

    assert calib.calibrated and calib.target == "card"
    assert measure(calib, CARD_PROBE, H_gt) == pytest.approx(CARD_PROBE_MM, abs=1.0)


def test_the_real_config_file_parses_into_settings():
    """Cheap, but it means a typo in config.yaml fails here rather than at 3am in
    front of a camera. Also pins the two facts about the shipped config that the
    rest of the project depends on."""
    from pipeline.config import load_config

    settings = CalibrationSettings.from_config(load_config())

    assert settings.card_width_mm == 85.60 and settings.card_height_mm == 53.98
    assert settings.marker_length_mm is None, (
        "config.yaml ships with no marker size on purpose — it depends on my "
        "phone's PPI and I haven't measured it"
    )


def test_card_detector_rejects_a_square_that_is_not_card_shaped():
    """The aspect-ratio filter is the only thing separating a card from any other
    bright rectangle. A square must not be accepted as a card — if it were, the
    scale would be silently wrong by a factor of 1.586."""
    q = syn.quad(700, 450, 500, 500)
    frame, _ = syn.card_scene(q, width_mm=85.60, height_mm=85.60)
    calib = calibrate(frame, card_settings())

    assert calib.calibrated is False
    assert "card" in calib.reason
