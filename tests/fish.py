"""Hand-built fish and hand-built calibrations, with answers I know in advance.

Same idea as tests/synthetic.py does for calibration: construct the input from
the answer, then check the code recovers the answer. Nothing here asserts.
"""

from __future__ import annotations

import numpy as np

from pipeline.calibrate import Calibration
from pipeline.landmarks import Landmarks

# A fish laid out in millimetres in its own frame, snout at the origin, tail to
# the +x side, dorsal to -y (image convention: y grows downward). Fork length is
# exactly 200 mm and body depth exactly 50 mm by construction, so any test can
# state its expected answer as a literal.
FISH_200MM: dict[str, tuple[float, float]] = {
    "snout_tip": (0.0, 0.0),
    "eye_anterior": (16.0, -6.0),
    "eye_posterior": (26.0, -6.0),
    "operculum_posterior": (48.0, 0.0),
    "dorsal_origin": (90.0, -25.0),
    "dorsal_insertion": (125.0, -21.0),
    "dorsal_apex": (105.0, -44.0),
    "ventral_margin": (90.0, 25.0),
    "peduncle_dorsal": (178.0, -9.6),
    "peduncle_ventral": (178.0, 9.6),
    "caudal_fork": (200.0, 0.0),
    "caudal_tip": (222.0, 0.0),
}

FORK_LENGTH_MM = 200.0
BODY_DEPTH_MM = 50.0
PEDUNCLE_DEPTH_MM = 19.2
DEPTH_RATIO = BODY_DEPTH_MM / FORK_LENGTH_MM  # 0.25
TOTAL_LENGTH_MM = 222.0


def scale_calibration(
    mm_per_px: float = 0.5, target_span_px: float = 400.0, residual_px: float = 0.0
) -> Calibration:
    """A calibration that is exactly a uniform scale, with no rotation and no tilt.

    Real calibrations aren't this clean, but measurement is not the place to test
    calibration; tests/test_calibrate.py already does that against rendered
    scenes. Here I want a transform whose answer I can compute in my head, so a
    failure means measure.py is wrong rather than something upstream.
    """
    H = np.array(
        [[mm_per_px, 0.0, 0.0], [0.0, mm_per_px, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    s = target_span_px
    corners = np.array([[0.0, 0.0], [s, 0.0], [s, s], [0.0, s]], dtype=np.float64)
    return Calibration(
        calibrated=True,
        target="synthetic",
        H=H,
        corners_px=corners,
        residual_px=residual_px,
        residual_meaningful=residual_px > 0.0,
        obliquity=1.0,
        mm_per_px=mm_per_px,
    )


def uncalibrated() -> Calibration:
    return Calibration(calibrated=False, reason="no target in this test")


def place(
    fish_mm: dict[str, tuple[float, float]] | None = None,
    mm_per_px: float = 0.5,
    origin_px: tuple[float, float] = (120.0, 340.0),
    angle_deg: float = 0.0,
    confidence: float | dict[str, float] = 0.9,
    bend_mm: float = 0.0,
    drop: tuple[str, ...] = (),
) -> Landmarks:
    """Put a millimetre fish into pixel space through the inverse of the scale.

    `bend_mm` bows the midline: each landmark is pushed perpendicular to the body
    axis by bend_mm * sin(pi * x/fork_length), zero at both ends and maximal in
    the middle. `drop` removes landmarks so the missing-landmark paths can be
    tested without inventing a broken detector.
    """
    fish = dict(fish_mm or FISH_200MM)
    for name in drop:
        fish.pop(name, None)

    t = np.deg2rad(angle_deg)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    o = np.array(origin_px, dtype=np.float64)

    out: dict[str, tuple[float, float]] = {}
    for name, (x, y) in fish.items():
        y = y + bend_mm * np.sin(np.pi * np.clip(x / FORK_LENGTH_MM, 0.0, 1.0))
        p = o + R @ (np.array([x, y]) / mm_per_px)
        out[name] = (float(p[0]), float(p[1]))
    return Landmarks.from_mapping(out, confidence=confidence, source="test")
