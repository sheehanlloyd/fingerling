"""Landmarks in, traits out, with an error bar on each one.

Everything here is first-order propagation of independent Gaussian errors. That
approximation is stated in full in `THE UNCERTAINTY MODEL` below, because an
error bar whose assumptions nobody wrote down is worse than no error bar — it
looks like knowledge.

Two structural choices worth knowing before you read the code:

  Distances are computed in pixels first and converted to millimetres second.
  So a frame with no calibration still produces every dimensionless trait
  (depth ratio, curvature index) at full quality, and only the millimetre traits
  go null. That's the right degradation: a fish can be culled for spinal
  curvature without any calibration target in frame at all.

  Which traits are computable is read off `TRAIT_REQUIREMENTS` in landmarks.py,
  not hardcoded here. A schema missing a landmark yields None plus a reason, not
  a substituted landmark.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pipeline.calibrate import Calibration
from pipeline.landmarks import TRAIT_REQUIREMENTS, Landmarks, midline_points

# ---------------------------------------------------------------------------
# THE UNCERTAINTY MODEL, in full
#
# 1. A landmark's positional error is isotropic Gaussian in the image plane, with
#    a standard deviation read off its confidence by straight-line interpolation
#    between `sigma_at_conf1_px` and `sigma_at_conf0_px`. That mapping is a
#    PLACEHOLDER: nobody has measured what a confidence of 0.7 is worth in pixels
#    on this model, because there is no model yet. Phase 2 can fit it from the
#    validation split; until then the shape of the curve is a guess and only the
#    ordering it produces is meaningful.
#
# 2. Landmark errors are INDEPENDENT of each other. This is the assumption most
#    likely to be wrong. A pose model that misses the whole fish by 5 px moves
#    every landmark together, and a common-mode offset like that cancels out of a
#    distance completely. So for correlated error this model OVER-estimates, and
#    the true error bar on a length is smaller than reported. I would rather be
#    wrong in that direction.
#
# 3. For a distance d = |b - a|, only the error component ALONG the a-b axis
#    changes d to first order; the perpendicular component is second-order small.
#    So sigma_d = sqrt(sigma_a^2 + sigma_b^2). This is a first-order expansion
#    and it degrades when sigma is comparable to d — i.e. for very short spans
#    like the peduncle on a small fish.
#
# 4. Calibration contributes a RELATIVE scale error, not an absolute one:
#    corner error / target span in pixels. That is why extrapolating a long fish
#    from a small target costs accuracy, and it's why the error bar on a 200 mm
#    length measured off a 40 mm marker is much wider than off a 200 mm ruler.
#    Where the card path gives a genuine over-determined residual that number is
#    used; the ArUco path can't produce one (see docs/CALIBRATION.md) so an
#    ASSUMED corner error from config stands in. That assumed value is a guess.
#
# 5. Scale error CANCELS in a ratio. depth_ratio and curvature_index therefore
#    carry no calibration term at all — only the landmark term. That is the
#    entire argument for preferring ratios, and it's why the deformity trait is
#    the dimensionless one.
#
# 6. NOT MODELLED, and larger than everything above: the coplanarity bias. The
#    fish sits above the calibration plane, so every length reads long by roughly
#    d / (d - h) for camera distance d and midline height h. It's systematic, it
#    doesn't shrink with averaging, and it is not in these error bars. See
#    docs/CALIBRATION.md.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MeasureSettings:
    sigma_at_conf1_px: float = 0.8
    sigma_at_conf0_px: float = 6.0
    assumed_corner_error_px: float = 0.5

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "MeasureSettings":
        m = cfg.get("measure", {})
        return cls(
            sigma_at_conf1_px=float(m.get("sigma_at_conf1_px", 0.8)),
            sigma_at_conf0_px=float(m.get("sigma_at_conf0_px", 6.0)),
            assumed_corner_error_px=float(m.get("assumed_corner_error_px", 0.5)),
        )


@dataclass(frozen=True)
class Quantity:
    """A number and its standard uncertainty, in stated units."""

    value: float
    sigma: float
    unit: str

    def __repr__(self) -> str:  # reads well in a REPL and in test failures
        return f"{self.value:.4g} ± {self.sigma:.2g} {self.unit}"

    def as_dict(self) -> dict[str, Any]:
        return {"value": self.value, "sigma": self.sigma, "unit": self.unit}


@dataclass
class MeasurementSet:
    """Every trait for one fish, plus why the missing ones are missing."""

    calibrated: bool
    fork_length_mm: Quantity | None = None
    total_length_mm: Quantity | None = None
    body_depth_mm: Quantity | None = None
    peduncle_depth_mm: Quantity | None = None
    depth_ratio: Quantity | None = None
    curvature_index: Quantity | None = None
    condition_factor: Quantity | None = None
    # Pixel-space equivalents, always present when the landmarks allow. These are
    # what the UI draws and what an uncalibrated frame reports instead of mm.
    fork_length_px: Quantity | None = None
    body_depth_px: Quantity | None = None
    peduncle_depth_px: Quantity | None = None
    weight_g: float | None = None
    unavailable: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"calibrated": self.calibrated}
        for name in (
            "fork_length_mm",
            "total_length_mm",
            "body_depth_mm",
            "peduncle_depth_mm",
            "depth_ratio",
            "curvature_index",
            "condition_factor",
            "fork_length_px",
            "body_depth_px",
            "peduncle_depth_px",
        ):
            q = getattr(self, name)
            out[name] = q.as_dict() if q is not None else None
        out["weight_g"] = self.weight_g
        out["unavailable"] = dict(self.unavailable)
        return out


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def landmark_sigma_px(conf: float, s: MeasureSettings) -> float:
    """Confidence 0..1 to a positional standard deviation in pixels."""
    c = float(np.clip(conf, 0.0, 1.0))
    return s.sigma_at_conf0_px + c * (s.sigma_at_conf1_px - s.sigma_at_conf0_px)


def _distance_px(lm: Landmarks, a: str, b: str, s: MeasureSettings) -> Quantity:
    """Pixel distance between two landmarks, with its first-order uncertainty."""
    pa, pb = lm.point(a), lm.point(b)
    d = float(np.hypot(*(pb - pa)))
    sa = landmark_sigma_px(lm.conf(a), s)
    sb = landmark_sigma_px(lm.conf(b), s)
    return Quantity(d, float(np.hypot(sa, sb)), "px")


def _target_span_px(calib: Calibration) -> float | None:
    """A single length scale for the calibration target, in pixels.

    sqrt of the corner quad's area — insensitive to which way round the corners
    are and to the target's aspect ratio, both of which vary.
    """
    if calib.corners_px is None:
        return None
    q = np.asarray(calib.corners_px, dtype=np.float64).reshape(-1, 2)
    x, y = q[:, 0], q[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    return float(np.sqrt(area)) if area > 0 else None


def relative_scale_sigma(calib: Calibration, s: MeasureSettings) -> float:
    """Fractional uncertainty in the pixel->mm scale factor.

    Corner error divided by the target's span in pixels. A 0.5 px error on a
    400 px marker is 0.125%; the same error on an 80 px marker is 0.625%, and
    every millimetre this frame produces inherits that.
    """
    span = _target_span_px(calib)
    if not span:
        return 0.0
    if calib.residual_meaningful and calib.residual_px is not None:
        corner_err = float(calib.residual_px)
    else:
        corner_err = s.assumed_corner_error_px
    return corner_err / span


def _to_mm(q_px: Quantity, calib: Calibration, s: MeasureSettings) -> Quantity | None:
    """Convert a pixel distance to millimetres by actually mapping the endpoints
    is not possible here — we only kept the scalar — so this uses mm_per_px.

    That's an approximation under perspective: mm_per_px varies across the frame.
    It's fine at low obliquity, which is the only regime `reliable` allows, and
    the exact path (`measure_traits` below maps the real endpoints through the
    homography) is what actually gets used for the traits. This helper only backs
    the uncertainty conversion.
    """
    if not calib.reliable or calib.mm_per_px is None:
        return None
    return Quantity(q_px.value * calib.mm_per_px, q_px.sigma * calib.mm_per_px, "mm")


def _distance_mm(
    lm: Landmarks, a: str, b: str, calib: Calibration, s: MeasureSettings
) -> Quantity | None:
    """Exact millimetre distance: map both endpoints through the homography, then
    attach the landmark term (converted through the local scale) and the
    calibration scale term in quadrature.

    Gated on `reliable`, not on `calibrated`, and the difference is the whole
    point. A frame can yield a homography and still be worthless. A real test
    frame with no card in it found something card-shaped at obliquity 4.57 with a
    21 px outline residual, and this function returned 84.42 mm for it. The row
    was correctly tagged unreliable and the millimetre figure was written to the
    database anyway, where the next person to read that column has no reason to
    doubt it. A number nobody should use must not exist, not merely travel with a
    flag that something else has to remember to check.
    """
    if not calib.reliable:
        return None
    d_mm = calib.distance_mm(lm.point(a), lm.point(b))
    px = _distance_px(lm, a, b, s)
    scale = calib.mm_per_px or 0.0
    sigma_landmark = px.sigma * scale
    sigma_scale = d_mm * relative_scale_sigma(calib, s)
    return Quantity(d_mm, float(np.hypot(sigma_landmark, sigma_scale)), "mm")


def _ratio(num: Quantity, den: Quantity) -> Quantity | None:
    """Ratio of two quantities in the SAME unit, so the scale factor cancels.

    Both inputs must be pixel quantities — that's deliberate. Taking the ratio in
    pixels means no calibration term enters at all, which is exactly the property
    that makes the ratio survive a bad calibration.
    """
    if den.value <= 0:
        return None
    v = num.value / den.value
    rel = np.hypot(num.sigma / num.value if num.value else 0.0, den.sigma / den.value)
    return Quantity(float(v), float(abs(v) * rel), "")


def curvature(lm: Landmarks, s: MeasureSettings) -> Quantity | None:
    """Spinal curvature index: max midline deviation from the snout-fork axis,
    divided by fork length. Dimensionless, so no calibration needed.

    Straight fish score near zero. What this MISSES: the midline is four derived
    points, so a fish bent between two of them is invisible to this. It's a
    coarse deformity proxy and calling it anything stronger would be a lie —
    see docs/DATASETS.md.
    """
    mid = midline_points(lm)
    if mid is None or not lm.has("snout_tip", "caudal_fork"):
        return None
    a = lm.point("snout_tip")
    b = lm.point("caudal_fork")
    axis = b - a
    L = float(np.hypot(*axis))
    if L < 1e-6:
        return None
    n = np.array([-axis[1], axis[0]]) / L  # unit normal to the axis

    dev = (mid - a) @ n  # signed perpendicular offsets
    t = ((mid - a) @ axis) / (L * L)  # position along the axis, 0..1
    i = int(np.argmax(np.abs(dev)))
    max_dev = float(abs(dev[i]))

    # Uncertainty on that deviation: the midline point's own scatter, plus the
    # wobble of the reference axis itself at that position along it.
    sig_mid = landmark_sigma_px(lm.mean_confidence(), s)
    sa = landmark_sigma_px(lm.conf("snout_tip"), s)
    sb = landmark_sigma_px(lm.conf("caudal_fork"), s)
    ti = float(np.clip(t[i], 0.0, 1.0))
    sigma_dev = float(
        np.sqrt(sig_mid**2 + ((1 - ti) * sa) ** 2 + (ti * sb) ** 2)
    )
    sigma_L = float(np.hypot(sa, sb))

    k = max_dev / L
    sigma_k = float(np.sqrt((sigma_dev / L) ** 2 + (max_dev * sigma_L / L**2) ** 2))
    return Quantity(k, sigma_k, "")


def condition_factor(fork_mm: Quantity, weight_g: float) -> Quantity:
    """Fulton's K = 100 * W / L^3, with W in grams and L in CENTIMETRES.

    The centimetre convention is what makes a healthy fish score near 1.0; do it
    in millimetres and every number comes out a thousand times smaller and
    nobody recognises it. Weight is taken as exact because it's typed in by hand
    and I have no scale spec to put an error bar on — so this uncertainty is the
    length contribution only, and it's an underestimate by however bad the scale
    is.
    """
    L_cm = fork_mm.value / 10.0
    k = 100.0 * weight_g / (L_cm**3)
    rel = 3.0 * (fork_mm.sigma / fork_mm.value) if fork_mm.value else 0.0
    return Quantity(float(k), float(abs(k) * rel), "")


# ---------------------------------------------------------------------------
# the entry point
# ---------------------------------------------------------------------------


def measure_traits(
    lm: Landmarks,
    calib: Calibration,
    settings: MeasureSettings,
    weight_g: float | None = None,
) -> MeasurementSet:
    """All traits for one detection. Never raises on missing landmarks."""
    out = MeasurementSet(calibrated=bool(calib.reliable), weight_g=weight_g)

    # Why millimetres are missing, if they are. "No target in frame" and "target
    # found but the view is too oblique to believe" are different problems with
    # different fixes, and a single message for both wastes the distinction.
    if calib.reliable:
        no_mm = None
    elif not calib.calibrated:
        no_mm = f"frame is not calibrated ({calib.reason or 'no target found'})"
    else:
        bits = []
        if calib.obliquity is not None and calib.obliquity > calib.max_obliquity:
            bits.append(f"obliquity {calib.obliquity:.2f} > {calib.max_obliquity}")
        if calib.residual_meaningful and calib.residual_px is not None \
                and calib.residual_px > calib.max_residual_px:
            bits.append(f"residual {calib.residual_px:.2f}px > {calib.max_residual_px}")
        no_mm = "frame is not calibrated well enough: " + (
            "; ".join(bits) or "calibration flagged unreliable"
        )

    if not lm.detected:
        out.unavailable = {t: "no detection" for t in TRAIT_REQUIREMENTS}
        return out

    def check(trait: str) -> bool:
        need = TRAIT_REQUIREMENTS[trait]
        missing = [n for n in need if not lm.has(n)]
        if missing:
            out.unavailable[trait] = "missing landmarks: " + ", ".join(missing)
            return False
        return True

    fork_px = depth_px = ped_px = None

    if check("fork_length_mm"):
        fork_px = _distance_px(lm, "snout_tip", "caudal_fork", settings)
        out.fork_length_px = fork_px
        out.fork_length_mm = _distance_mm(lm, "snout_tip", "caudal_fork", calib, settings)
        if out.fork_length_mm is None:
            out.unavailable["fork_length_mm"] = no_mm

    if check("total_length_mm"):
        out.total_length_mm = _distance_mm(lm, "snout_tip", "caudal_tip", calib, settings)
        if out.total_length_mm is None:
            out.unavailable["total_length_mm"] = no_mm

    if check("body_depth_mm"):
        depth_px = _distance_px(lm, "dorsal_origin", "ventral_margin", settings)
        out.body_depth_px = depth_px
        out.body_depth_mm = _distance_mm(
            lm, "dorsal_origin", "ventral_margin", calib, settings
        )
        if out.body_depth_mm is None:
            out.unavailable["body_depth_mm"] = no_mm

    if check("peduncle_depth_mm"):
        ped_px = _distance_px(lm, "peduncle_dorsal", "peduncle_ventral", settings)
        out.peduncle_depth_px = ped_px
        out.peduncle_depth_mm = _distance_mm(
            lm, "peduncle_dorsal", "peduncle_ventral", calib, settings
        )
        if out.peduncle_depth_mm is None:
            out.unavailable["peduncle_depth_mm"] = no_mm

    # Ratio is taken in pixels on purpose — see _ratio. No calibration needed.
    if check("depth_ratio") and depth_px is not None and fork_px is not None:
        out.depth_ratio = _ratio(depth_px, fork_px)

    if check("curvature_index"):
        out.curvature_index = curvature(lm, settings)
        if out.curvature_index is None:
            out.unavailable["curvature_index"] = "midline could not be built"

    if weight_g is not None and out.fork_length_mm is not None:
        out.condition_factor = condition_factor(out.fork_length_mm, weight_g)
    elif weight_g is None:
        out.unavailable["condition_factor"] = "no weight entered"
    else:
        out.unavailable["condition_factor"] = "needs a calibrated fork length"

    return out
