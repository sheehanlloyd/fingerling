"""Pixels to millimetres.

An object whose physical size I know exactly sits in the frame. Find its corners,
solve a homography from the image plane to that object's plane, and then any point
on that plane converts to millimetres.

Two targets:

  ArUco marker on my phone screen. I pick the pixel size that gives an exact
  physical size from the screen's PPI, so there's no printer scaling error.

  Credit card. ISO/IEC 7810 ID-1 is 85.60 x 53.98 mm worldwide, so I know its
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
   segmentation, none of which the 4-corner solve can see.

2. WHAT ACTUALLY CATCHES A BAD VIEW IS OBLIQUITY, NOT RESIDUAL.
   A homography from a steeply tilted plane stretches the image far more in one
   direction than the other, and stretches it differently at different points.
   Both are readable straight off the matrix. `obliquity` is the worst of those
   two effects across the target's footprint. Square-on it's 1.0. In-plane
   rotation is a rigid motion so it stays 1.0. Edge-on it blows up. That is the
   number that flags a bad frame.

Coplanarity: I assume the target and the fish sit on the same plane. They don't,
a fish has thickness and its midline sits above the board by roughly half its
body depth. That's a systematic magnification error, not noise, and it makes
every length read slightly long. See docs/CALIBRATION.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

# ISO/IEC 7810 ID-1. Not a measurement, the standard itself.
CARD_ID1_WIDTH_MM = 85.60
CARD_ID1_HEIGHT_MM = 53.98
# ID-1 also specifies the corner radius, and it matters: the outline residual has
# to ignore the arcs or every real card fails the check. See _outline_residual_px.
CARD_CORNER_RADIUS_MM = 3.18


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
    all any measurement here needs, because every trait is a distance or a ratio.
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
    contour_px: np.ndarray,
    H: np.ndarray,
    rect_mm: np.ndarray,
    corner_exclusion_mm: float = CARD_CORNER_RADIUS_MM * 1.6,
) -> float:
    """RMS distance, in pixels, from the detected outline to the ideal rectangle.

    This is the over-determined check the 4-corner solve can't give me. Map every
    outline point into the board plane, measure how far it lands from the true
    rectangle's edges, and divide by the local scale to get back to pixels so the
    threshold means something in image terms.

    CORNERS ARE EXCLUDED, and they have to be. ID-1 specifies a corner radius of
    3.18 mm, so a real card's outline is four straight edges joined by four arcs.
    Measured against a sharp-cornered rectangle each arc departs from it by about
    r(1 - 1/sqrt(2)) ~ 0.93 mm, which is enormous next to the sub-pixel deviation
    this metric exists to detect.

    I only found this when I photographed an actual card: residual 17.5 px on a
    frame that was otherwise fine, which flagged a perfectly good calibration as
    unreliable and withheld every millimetre. My synthetic scenes all render
    sharp-cornered rectangles, so none of them could ever have caught it.

    Excluding a margin of 1.6x the corner radius leaves the straight edges, which
    is what the check is actually about. A bent card or a distorting lens bows
    the EDGES, and that's still measured.
    """
    pts_px = np.asarray(contour_px, dtype=np.float64).reshape(-1, 2)
    pts_mm = cv2.perspectiveTransform(pts_px.reshape(-1, 1, 2), H).reshape(-1, 2)

    corner_dist = np.min(
        np.linalg.norm(pts_mm[:, None, :] - rect_mm[None, :, :], axis=2), axis=1
    )
    keep = corner_dist > corner_exclusion_mm
    if keep.sum() < 16:
        return float("inf")  # nothing left but corners; the outline is not a card
    pts_px, pts_mm = pts_px[keep], pts_mm[keep]

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

    If several markers are in frame I take the biggest one and ignore the rest,
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
    # 520..879, one pixel short, an inward bias of about 0.3% on a 360 px
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


def _corners_from_edges(
    frame: np.ndarray,
    quad: np.ndarray,
    outline: np.ndarray,
    corner_radius_mm: float = CARD_CORNER_RADIUS_MM,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Recover the card's true sharp corners by fitting its four straight edges.

    THIS IS THE BIGGEST ACCURACY FIX IN THE FILE and it took real photographs to
    find. A card has rounded corners (ID-1 specifies a 3.18 mm radius) so
    `approxPolyDP` returns four vertices that sit ON the arcs, not where the
    edges would meet if extended. Each vertex is inset from the true corner by
    about r(sqrt(2) - 1) along the diagonal, which is 0.293r perpendicular to
    each edge: 0.93 mm for an ID-1 card.

    That is not a cosmetic problem. The solver is told those four points span
    85.60 mm when they really span 85.60 - 2(0.93) = 83.74 mm of card, so every
    millimetre it produces afterwards is too big by 85.60/83.74 = 2.2%.

    How I found it: on four real photographs, ZERO of 6,460 outline points fell
    inside the ideal rectangle. Every single one was outside it, by a median of
    0.929 mm. A uniform one-sided offset is what an inset corner looks like; a
    bent card or a bad lens would scatter to both sides. Measured over-read on
    those shots was +2.9%, against +2.2% predicted from the geometry alone.

    The fix is the standard one: throw away the arcs, fit a line through each
    straight edge, and intersect consecutive lines. The corners come from
    hundreds of points each instead of one, so this is also less noisy than the
    sub-pixel nudge it replaces.

    The lines are NOT fitted to the contour points. A contour is wherever the
    segmentation happened to put it, and `_card_candidates` dilates its Canny
    mask, which pushes the traced outline about 2 px outward, a bias that
    `cornerSubPix` used to hide and that this function would otherwise inherit.
    So each contour point is first pushed along the edge normal onto the peak of
    the intensity gradient, with a parabolic sub-pixel fit, and the line is
    fitted to those. That reads the edge off the photograph rather than off a
    threshold, so it doesn't matter how the mask was made.

    Returns (corners, edge_points) or None if any edge has too few points to fit,
    in which case the caller keeps the original corners. The edge points come
    back because the residual has to be measured against the same thing the
    corners were fitted to. Measuring a gradient-fitted rectangle against a
    mask-derived contour just re-measures how much the mask was dilated.
    """
    pts = np.asarray(outline, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 64:
        return None

    # Rough scale from the quad itself: good enough to know how much of each
    # edge is arc, which is all it's used for.
    side_a = (np.linalg.norm(quad[0] - quad[1]) + np.linalg.norm(quad[3] - quad[2])) / 2
    side_b = (np.linalg.norm(quad[0] - quad[3]) + np.linalg.norm(quad[1] - quad[2])) / 2
    long_px, short_px = max(side_a, side_b), min(side_a, side_b)
    if short_px < 1e-6:
        return None
    px_per_mm = long_px / max(CARD_ID1_WIDTH_MM, CARD_ID1_HEIGHT_MM)
    exclude_px = corner_radius_mm * px_per_mm * 1.8

    grey = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grey = cv2.GaussianBlur(grey.astype(np.float32), (5, 5), 0)
    gx = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)

    def sample(p: np.ndarray) -> np.ndarray:
        return cv2.remap(
            grad,
            p[:, 0].astype(np.float32).reshape(-1, 1),
            p[:, 1].astype(np.float32).reshape(-1, 1),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        ).ravel()

    search_px = 4.0
    offsets = np.arange(-search_px, search_px + 0.5, 0.5)

    lines = []
    refined: list[np.ndarray] = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        ab = b - a
        n = float(np.hypot(*ab))
        if n < 1e-6:
            return None
        u = ab / n
        nrm = np.array([-u[1], u[0]])
        t = (pts - a) @ u                       # position along the edge
        perp = np.abs((pts - a) @ nrm)
        on_edge = (
            (t > exclude_px) & (t < n - exclude_px)   # not on either arc
            & (perp < 0.04 * n)                       # actually near this edge
        )
        if on_edge.sum() < 16:
            return None
        e = pts[on_edge]

        # Push each point onto the gradient ridge along the normal.
        profiles = np.stack([sample(e + nrm * d) for d in offsets], axis=1)
        k = profiles.argmax(axis=1)
        interior = (k > 0) & (k < len(offsets) - 1)
        shift = offsets[k].astype(np.float64)
        if interior.any():
            ki = k[interior]
            rows = np.flatnonzero(interior)
            y0 = profiles[rows, ki - 1]
            y1 = profiles[rows, ki]
            y2 = profiles[rows, ki + 1]
            denom = y0 - 2.0 * y1 + y2
            frac = np.where(np.abs(denom) < 1e-9, 0.0, 0.5 * (y0 - y2) / denom)
            shift[rows] = offsets[ki] + np.clip(frac, -1.0, 1.0) * 0.5
        e = e + nrm * shift[:, None]
        refined.append(e)

        # Total least squares: the principal direction of the edge points.
        c = e.mean(axis=0)
        _, _, vt = np.linalg.svd(e - c, full_matrices=False)
        lines.append((c, vt[0]))

    corners = []
    for i in range(4):
        (p1, d1), (p2, d2) = lines[i - 1], lines[i]
        A = np.array([d1, -d2]).T
        det = float(np.linalg.det(A))
        if abs(det) < 1e-9:
            return None                          # parallel edges: not a quad
        st = np.linalg.solve(A, p2 - p1)
        corners.append(p1 + st[0] * d1)
    corners = np.array(corners, dtype=np.float64)

    # Sanity: the fitted corners must be near the ones we started from. If the
    # fit has wandered further than a corner radius or two, something else got
    # fitted and the original quad is the safer answer.
    if np.max(np.linalg.norm(corners - quad, axis=1)) > 4.0 * corner_radius_mm * px_per_mm:
        return None
    return corners, np.vstack(refined)


def _refine_corners(frame: np.ndarray, quad: np.ndarray, max_shift_px: float = 4.0) -> np.ndarray:
    """Nudge integer contour corners onto the actual intensity corner.

    approxPolyDP picks its vertices from a binarised contour, so they land on
    whole pixels and inherit the same inward bias the ArUco corners had. This
    costs a few lines and buys back most of it. If the refinement wanders further
    than max_shift_px it's found something else (a texture corner on the card
    face, say) so keep the original.
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
    rectangle in frame, so it's doing real work. It's deliberately loose,
    perspective changes the apparent ratio, and tightening it would reject good
    frames more often than it rejects wrong objects.

    Two filters here exist because a test caught them, not because I planned them:
    a quad touching the image border is rejected (the frame's own edge is a
    perfect rectangle, and on a 1400x900 canvas its aspect ratio is within 2% of
    a credit card's, and the detector cheerfully calibrated against the whole frame),
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

        # Prefer corners recovered by fitting the four straight edges: a card's
        # corners are rounded, so approxPolyDP's vertices sit on the arcs and are
        # inset by about 0.93 mm, which becomes a 2.2% scale error. Fall back to
        # the sub-pixel nudge when the edge fit can't run (too few points, a
        # partially occluded edge).
        outline = contour.reshape(-1, 2).astype(np.float64)
        fitted = _corners_from_edges(frame, quad, outline)
        if fitted is not None:
            quad, outline = fitted
        else:
            quad = _refine_corners(frame, quad)

        # Average the two opposite sides. Under perspective they differ, and the
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
        best = (quad, board, outline)
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
    """Try to calibrate one frame. Never raises on a missing or unusable target.
    It comes back uncalibrated with a reason, and the caller reports pixels."""
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
                    "aruco: marker_length_mm is null in config, refusing to invent a scale"
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
