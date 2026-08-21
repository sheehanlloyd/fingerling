"""Pixels to millimetres.

An object whose physical size I know exactly sits in the frame. Find its corners,
solve a homography from the image plane to that object's plane, and then any point
on that plane converts to millimetres.

Two targets:

  ArUco marker on my phone screen — I pick the pixel size that gives an exact
  physical size from the screen's PPI, so there's no printer scaling error.

  Credit card — ISO/IEC 7810 ID-1 is 85.60 x 53.98 mm worldwide, so I know its
  size to a hundredth of a millimetre without owning a caliper.

Two things I want you to read before trusting anything this module reports:

1. REPROJECTION RESIDUAL IS NOT A FREE LUNCH ON A 4-POINT SOLVE.
   A homography has 8 degrees of freedom. Four point correspondences give exactly
   8 equations. The solution is exact and the forward reprojection error is zero
   to floating-point noise *no matter how badly the target is positioned*. So for
   a single ArUco marker, "residual" is a number that is always ~0 and tells you
   nothing. I report it anyway, with `residual_meaningful=False` set, so nobody
   reads it as a quality signal.

   For the card I do get a real residual, because I don't only use the four
   corners: I take every point on the detected outline, map it to the board plane,
   and measure how far it sits from the ideal rectangle's edges. That is an
   over-determined check. It catches lens distortion, a bent card, and sloppy
   segmentation — none of which the 4-corner solve can see.

2. WHAT ACTUALLY CATCHES A BAD VIEW IS OBLIQUITY, NOT RESIDUAL.
   A homography from a steeply tilted plane stretches the image far more in one
   direction than the other, and stretches it differently at different points.
   Both are readable straight off the matrix. `obliquity` is the worst of those
   two effects across the target's footprint. Square-on it's 1.0. In-plane
   rotation is a rigid motion so it stays 1.0. Edge-on it blows up. That is the
   number that flags a bad frame.

Coplanarity: I assume the target and the fish sit on the same plane. They don't —
a fish has thickness and its midline sits above the board by roughly half its
body depth. That's a systematic magnification error, not noise, and it makes
every length read slightly long. See docs/CALIBRATION.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

# ISO/IEC 7810 ID-1. Not a measurement — the standard itself.
CARD_ID1_WIDTH_MM = 85.60
CARD_ID1_HEIGHT_MM = 53.98


class CalibrationUnavailable(RuntimeError):
    """Raised when someone asks an uncalibrated frame for millimetres."""


@dataclass(frozen=True)
class CalibrationSettings:
    """Everything calibration needs, pulled out of config.yaml so the functions
    below never read global state and are trivial to test."""

    target: str = "auto"  # auto | aruco | card | none
    aruco_dictionary: str = "DICT_4X4_50"
    marker_length_mm: float | None = None
    card_width_mm: float = CARD_ID1_WIDTH_MM
    card_height_mm: float = CARD_ID1_HEIGHT_MM
    card_aspect_tolerance: float = 0.18
    card_min_area_frac: float = 0.002
    card_max_area_frac: float = 0.5
    max_residual_px: float = 2.0
    max_obliquity: float = 2.0

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "CalibrationSettings":
        c = cfg["calibration"]
        aruco = c.get("aruco", {})
        card = c.get("card", {})
        return cls(
            target=c.get("target", "auto"),
            aruco_dictionary=aruco.get("dictionary", "DICT_4X4_50"),
            marker_length_mm=aruco.get("marker_length_mm"),
            card_width_mm=card.get("width_mm", CARD_ID1_WIDTH_MM),
            card_height_mm=card.get("height_mm", CARD_ID1_HEIGHT_MM),
            card_aspect_tolerance=card.get("aspect_tolerance", 0.18),
            card_min_area_frac=card.get("min_area_frac", 0.002),
            card_max_area_frac=card.get("max_area_frac", 0.5),
            max_residual_px=c.get("max_reprojection_residual_px", 2.0),
            max_obliquity=c.get("max_obliquity", 2.0),
        )


@dataclass(frozen=True)
class Calibration:
    """The result of trying to calibrate one frame.

    `H` maps image pixels to board-plane millimetres. Board-plane origin is the
    target's own top-left corner; the axes are the target's edges. That means
    millimetre coordinates are only comparable within a single frame, which is
    all any measurement here needs — every trait is a distance or a ratio.
    """

    calibrated: bool
    target: str | None = None
    H: np.ndarray | None = None
    corners_px: np.ndarray | None = None
    residual_px: float | None = None
    residual_meaningful: bool = False
    obliquity: float | None = None
    mm_per_px: float | None = None
    reason: str | None = None
    max_residual_px: float = 2.0
    max_obliquity: float = 2.0

    @property
    def reliable(self) -> bool:
        """Calibrated *and* the frame's geometry is good enough to believe.

        Note the `residual_meaningful` guard: an ArUco solve always reports ~0
        residual, so passing that gate proves nothing and only the obliquity gate
        does real work there.
        """
        if not self.calibrated:
            return False
        if self.obliquity is None or self.obliquity > self.max_obliquity:
            return False
        if self.residual_meaningful:
            if self.residual_px is None or self.residual_px > self.max_residual_px:
                return False
        return True

    @property
    def residual_mm(self) -> float | None:
        """Residual expressed in millimetres at the target's scale. This is what
        feeds measurement uncertainty."""
        if self.residual_px is None or self.mm_per_px is None:
            return None
        return self.residual_px * self.mm_per_px

    def to_mm(self, points_px: np.ndarray) -> np.ndarray:
        """Map (N, 2) image points to (N, 2) board-plane millimetres.

        Raises rather than returning something plausible-looking if there's no
        calibration. A silent fallback to pixels here would let pixel numbers
        leak into a field labelled millimetres, which is the exact failure this
        whole project is meant to avoid.
        """
        if not self.calibrated or self.H is None:
            raise CalibrationUnavailable(self.reason or "no calibration for this frame")
        pts = np.asarray(points_px, dtype=np.float64).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(pts, self.H)
        return out.reshape(-1, 2)

    def distance_mm(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Straight-line distance between two image points, in millimetres."""
        a, b = self.to_mm(np.array([p1, p2], dtype=np.float64))
        return float(np.hypot(*(b - a)))


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------


def _local_jacobian(H: np.ndarray, x: float, y: float) -> np.ndarray:
    """The 2x2 linear map that H looks like in a small neighbourhood of (x, y).

    Differentiating u = (h00 x + h01 y + h02) / d, with d = h20 x + h21 y + h22,
    gives du/dx = (h00 - u h20) / d and so on. Units: millimetres per pixel.
    """
    d = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    if abs(d) < 1e-12:
        return np.full((2, 2), np.inf)
    u = (H[0, 0] * x + H[0, 1] * y + H[0, 2]) / d
    v = (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / d
    return np.array(
        [
            [H[0, 0] - u * H[2, 0], H[0, 1] - u * H[2, 1]],
            [H[1, 0] - v * H[2, 0], H[1, 1] - v * H[2, 1]],
        ]
    ) / d


def plane_metrics(H: np.ndarray, sample_px: np.ndarray) -> tuple[float, float]:
    """Return (obliquity, mm_per_px) for a homography, sampled over some points.

    At each sample point take the singular values of the local Jacobian: they are
    the millimetres-per-pixel scale along the two directions that get stretched
    most and least.

      anisotropy  = worst s_max / s_min at any single point. How badly the map
                    stretches one direction relative to the other, i.e. tilt.
      scale_spread = largest mean scale anywhere / smallest anywhere. How much
                    the scale changes across the target, i.e. perspective.

    obliquity is the worse of the two. A square-on view and a purely rotated view
    both give exactly 1.0, because rotation is a rigid motion of the plane.
    mm_per_px is the mean scale, reported for uncertainty bookkeeping.
    """
    anis: list[float] = []
    scales: list[float] = []
    for x, y in np.asarray(sample_px, dtype=np.float64).reshape(-1, 2):
        J = _local_jacobian(H, float(x), float(y))
        if not np.isfinite(J).all():
            return float("inf"), float("nan")
        s = np.linalg.svd(J, compute_uv=False)
        if s[1] <= 1e-12:
            return float("inf"), float("nan")
        anis.append(float(s[0] / s[1]))
        scales.append(float(np.sqrt(s[0] * s[1])))
    spread = max(scales) / min(scales)
    return max(max(anis), spread), float(np.mean(scales))


def _order_quad(pts: np.ndarray) -> np.ndarray:
    """Sort 4 points into top-left, top-right, bottom-right, bottom-left.

    Sorting by angle about the centroid gives a consistent cycle (clockwise on
    screen, since y points down); rolling so the smallest x+y comes first picks
    the top-left as the start. Works for rotated quads, unlike the usual
    min/max-of-sums-and-differences trick.
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    c = pts.mean(axis=0)
    order = np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))
    pts = pts[order]
    start = int(np.argmin(pts.sum(axis=1)))
    return np.roll(pts, -start, axis=0)


def _point_to_rect_distance_mm(pts_mm: np.ndarray, rect_mm: np.ndarray) -> np.ndarray:
    """Shortest distance from each point to the boundary of a rectangle, in mm."""
    pts = np.asarray(pts_mm, dtype=np.float64).reshape(-1, 2)
    best = np.full(len(pts), np.inf)
    for i in range(4):
        a = rect_mm[i]
        b = rect_mm[(i + 1) % 4]
        ab = b - a
        denom = float(ab @ ab)
        if denom < 1e-12:
            continue
        t = np.clip(((pts - a) @ ab) / denom, 0.0, 1.0)
        proj = a + t[:, None] * ab
        best = np.minimum(best, np.linalg.norm(pts - proj, axis=1))
    return best


def _outline_residual_px(
    contour_px: np.ndarray, H: np.ndarray, rect_mm: np.ndarray
) -> float:
    """RMS distance, in pixels, from the detected outline to the ideal rectangle.

    This is the over-determined check the 4-corner solve can't give me. Map every
    outline point into the board plane, measure how far it lands from the true
    rectangle's edges, and divide by the local scale to get back to pixels so the
    threshold means something in image terms.
    """
    pts_px = np.asarray(contour_px, dtype=np.float64).reshape(-1, 2)
    pts_mm = cv2.perspectiveTransform(pts_px.reshape(-1, 1, 2), H).reshape(-1, 2)
    dist_mm = _point_to_rect_distance_mm(pts_mm, rect_mm)

    scales = np.empty(len(pts_px))
    for i, (x, y) in enumerate(pts_px):
        J = _local_jacobian(H, float(x), float(y))
        if not np.isfinite(J).all():
            return float("inf")
        s = np.linalg.svd(J, compute_uv=False)
        scales[i] = np.sqrt(max(s[0] * s[1], 1e-18))

    dist_px = dist_mm / scales
    return float(np.sqrt(np.mean(dist_px**2)))


# --------------------------------------------------------------------------
# target detection
# --------------------------------------------------------------------------


def find_aruco(
    frame: np.ndarray, marker_length_mm: float, dictionary: str = "DICT_4X4_50"
) -> tuple[np.ndarray, np.ndarray] | None:
    """Find the largest ArUco marker. Returns (corners_px, board_mm) or None.

    If several markers are in frame I take the biggest one and ignore the rest —
    without a known layout for the others there's no way to place them in a
    common board frame, and guessing one would be worse than dropping them.
    """
    if not hasattr(cv2.aruco, dictionary):
        raise ValueError(f"unknown ArUco dictionary {dictionary!r}")
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    params = cv2.aruco.DetectorParameters()
    # Subpixel refinement is off by default, and leaving it off costs real
    # accuracy. Unrefined, the detector reports the index of the outermost black
    # pixel, so a marker whose true edges span 519.5..879.5 comes back as
    # 520..879 — one pixel short, an inward bias of about 0.3% on a 360 px
    # marker. That bias scales straight through to millimetres and it's
    # systematic, so averaging frames won't remove it. With refinement the same
    # marker measures within about 0.03%.
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary)), params
    )
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(corners) == 0:
        return None

    biggest = max(corners, key=lambda c: abs(cv2.contourArea(c.reshape(4, 2))))
    pts = np.asarray(biggest, dtype=np.float64).reshape(4, 2)

    # cv2.aruco hands back corners in the marker's own order: top-left,
    # top-right, bottom-right, bottom-left. So the board coordinates follow
    # directly, no ordering step needed.
    L = float(marker_length_mm)
    board = np.array([[0.0, 0.0], [L, 0.0], [L, L], [0.0, L]], dtype=np.float64)
    return pts, board


def _card_candidates(frame: np.ndarray) -> list[np.ndarray]:
    """Closed contours that might be a card, from two different segmentations.

    Canny handles a card on a busy background; Otsu handles a flat, evenly lit
    one where the edges are weak. Running both and pooling the candidates is
    cheaper than trying to decide up front which situation I'm in.
    """
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    masks = []
    edges = cv2.Canny(gray, 50, 150)
    masks.append(cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1))
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    masks.append(otsu)
    masks.append(cv2.bitwise_not(otsu))

    out: list[np.ndarray] = []
    for m in masks:
        found, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        out.extend(found)
    return out


def _refine_corners(frame: np.ndarray, quad: np.ndarray, max_shift_px: float = 4.0) -> np.ndarray:
    """Nudge integer contour corners onto the actual intensity corner.

    approxPolyDP picks its vertices from a binarised contour, so they land on
    whole pixels and inherit the same inward bias the ArUco corners had. This
    costs a few lines and buys back most of it. If the refinement wanders further
    than max_shift_px it's found something else — a texture corner on the card
    face, say — so keep the original.
    """
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    seed = quad.astype(np.float32).reshape(-1, 1, 2)
    refined = cv2.cornerSubPix(
        gray,
        seed.copy(),
        winSize=(5, 5),
        zeroZone=(-1, -1),
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01),
    ).reshape(4, 2).astype(np.float64)

    shift = np.linalg.norm(refined - quad, axis=1)
    out = quad.copy()
    keep = shift <= max_shift_px
    out[keep] = refined[keep]
    return out


def find_card(
    frame: np.ndarray,
    width_mm: float = CARD_ID1_WIDTH_MM,
    height_mm: float = CARD_ID1_HEIGHT_MM,
    aspect_tolerance: float = 0.18,
    min_area_frac: float = 0.002,
    max_area_frac: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Find a card-shaped quad. Returns (corners_px, board_mm, outline_px) or None.

    The aspect ratio filter is the only thing separating a card from any other
    rectangle in frame, so it's doing real work. It's deliberately loose —
    perspective changes the apparent ratio, and tightening it would reject good
    frames more often than it rejects wrong objects.

    Two filters here exist because a test caught them, not because I planned them:
    a quad touching the image border is rejected (the frame's own edge is a
    perfect rectangle, and on a 1400x900 canvas its aspect ratio is within 2% of
    a credit card's — the detector cheerfully calibrated against the whole frame),
    and so is anything covering more than max_area_frac of the image. A card that
    runs off the edge has at least one corner in the wrong place anyway.
    """
    h, w = frame.shape[:2]
    min_area = min_area_frac * h * w
    max_area = max_area_frac * h * w
    target_ratio = max(width_mm, height_mm) / min(width_mm, height_mm)
    border = 2.0

    best = None
    best_area = 0.0
    for contour in _card_candidates(frame):
        area = abs(cv2.contourArea(contour))
        if area < min_area or area > max_area or area <= best_area:
            continue
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        quad = _order_quad(approx.reshape(4, 2))
        if (
            quad[:, 0].min() < border
            or quad[:, 1].min() < border
            or quad[:, 0].max() > w - 1 - border
            or quad[:, 1].max() > h - 1 - border
        ):
            continue

        quad = _refine_corners(frame, quad)

        # Average the two opposite sides — under perspective they differ, and the
        # mean is a better stand-in for the true edge length than either one.
        side_top_bottom = (
            np.linalg.norm(quad[0] - quad[1]) + np.linalg.norm(quad[3] - quad[2])
        ) / 2
        side_left_right = (
            np.linalg.norm(quad[0] - quad[3]) + np.linalg.norm(quad[1] - quad[2])
        ) / 2
        if min(side_top_bottom, side_left_right) < 1e-6:
            continue
        ratio = max(side_top_bottom, side_left_right) / min(
            side_top_bottom, side_left_right
        )
        if abs(ratio - target_ratio) / target_ratio > aspect_tolerance:
            continue

        long_mm, short_mm = max(width_mm, height_mm), min(width_mm, height_mm)
        if side_top_bottom >= side_left_right:
            bw, bh = long_mm, short_mm  # landscape in the image
        else:
            bw, bh = short_mm, long_mm  # portrait
        board = np.array(
            [[0.0, 0.0], [bw, 0.0], [bw, bh], [0.0, bh]], dtype=np.float64
        )
        best = (quad, board, contour.reshape(-1, 2).astype(np.float64))
        best_area = area

    return best


# --------------------------------------------------------------------------
# the entry point
# --------------------------------------------------------------------------


def _uncalibrated(reason: str, settings: CalibrationSettings) -> Calibration:
    return Calibration(
        calibrated=False,
        reason=reason,
        max_residual_px=settings.max_residual_px,
        max_obliquity=settings.max_obliquity,
    )


def calibrate(frame: np.ndarray, settings: CalibrationSettings) -> Calibration:
    """Try to calibrate one frame. Never raises on a missing or unusable target —
    it comes back uncalibrated with a reason, and the caller reports pixels."""
    if settings.target == "none":
        return _uncalibrated("calibration disabled in config", settings)

    order = {
        "auto": ("aruco", "card"),
        "aruco": ("aruco",),
        "card": ("card",),
    }.get(settings.target)
    if order is None:
        return _uncalibrated(f"unknown calibration target {settings.target!r}", settings)

    reasons: list[str] = []
    for kind in order:
        if kind == "aruco":
            if settings.marker_length_mm is None:
                reasons.append(
                    "aruco: marker_length_mm is null in config — refusing to invent a scale"
                )
                continue
            hit = find_aruco(frame, settings.marker_length_mm, settings.aruco_dictionary)
            if hit is None:
                reasons.append("aruco: no marker found")
                continue
            corners, board = hit
            outline = None
        else:
            hit = find_card(
                frame,
                settings.card_width_mm,
                settings.card_height_mm,
                settings.card_aspect_tolerance,
                settings.card_min_area_frac,
                settings.card_max_area_frac,
            )
            if hit is None:
                reasons.append("card: no card-shaped quad found")
                continue
            corners, board, outline = hit

        H, _ = cv2.findHomography(corners, board, method=0)
        if H is None:
            reasons.append(f"{kind}: homography solve failed")
            continue

        obliquity, mm_per_px = plane_metrics(
            H, np.vstack([corners, corners.mean(axis=0, keepdims=True)])
        )

        if outline is not None and len(outline) >= 8:
            residual = _outline_residual_px(outline, H, board)
            meaningful = True
        else:
            # Four correspondences, eight unknowns: this is ~0 by construction.
            back = cv2.perspectiveTransform(
                board.reshape(-1, 1, 2), np.linalg.inv(H)
            ).reshape(-1, 2)
            residual = float(np.sqrt(np.mean(np.sum((back - corners) ** 2, axis=1))))
            meaningful = False

        return Calibration(
            calibrated=True,
            target=kind,
            H=H,
            corners_px=corners,
            residual_px=residual,
            residual_meaningful=meaningful,
            obliquity=obliquity,
            mm_per_px=mm_per_px,
            reason=None,
            max_residual_px=settings.max_residual_px,
            max_obliquity=settings.max_obliquity,
        )

    return _uncalibrated("; ".join(reasons) or "no target found", settings)
