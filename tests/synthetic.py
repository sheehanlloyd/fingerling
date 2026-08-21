"""Synthetic calibration scenes with ground truth I control exactly.

The idea in every test: I define a board plane in millimetres, pick a homography
that maps that plane onto the image, and render the target through it. Then I
take points whose millimetre coordinates I chose myself, push them through the
same homography to get pixels, and hand those pixels to the calibration code. If
it gives me back the millimetres I started with, the math is right. If it doesn't,
the test fails with a number I can interpret.

Nothing here asserts. It only builds scenes.
"""

from __future__ import annotations

import cv2
import numpy as np

CANVAS_H, CANVAS_W = 900, 1400


def quad(cx: float, cy: float, w: float, h: float) -> np.ndarray:
    """Axis-aligned quad, corners in TL, TR, BR, BL order."""
    return np.array(
        [
            [cx - w / 2, cy - h / 2],
            [cx + w / 2, cy - h / 2],
            [cx + w / 2, cy + h / 2],
            [cx - w / 2, cy + h / 2],
        ],
        dtype=np.float64,
    )


def rotate(q: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate a quad about its own centre. A rigid motion — no new perspective."""
    c = q.mean(axis=0)
    t = np.deg2rad(degrees)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return (q - c) @ R.T + c


def foreshorten(q: np.ndarray, top_scale: float, height_scale: float = 1.0) -> np.ndarray:
    """Squeeze the top edge toward its midpoint and optionally flatten the whole
    quad. That's what a plane tilted away from the camera looks like: the far edge
    is shorter than the near one. top_scale of 1.0 is no tilt; 0.3 is steep."""
    q = q.copy()
    top_mid = (q[0] + q[1]) / 2
    q[0] = top_mid + (q[0] - top_mid) * top_scale
    q[1] = top_mid + (q[1] - top_mid) * top_scale
    c = q.mean(axis=0)
    q[:, 1] = c[1] + (q[:, 1] - c[1]) * height_scale
    return q


def board_to_image(board_quad_mm: np.ndarray, image_quad_px: np.ndarray) -> np.ndarray:
    """Ground-truth homography: board millimetres -> image pixels."""
    return cv2.getPerspectiveTransform(
        board_quad_mm.astype(np.float32), image_quad_px.astype(np.float32)
    )


def project(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def _render(src: np.ndarray, src_quad: np.ndarray, dst_quad: np.ndarray, bg: int) -> np.ndarray:
    M = cv2.getPerspectiveTransform(
        src_quad.astype(np.float32), dst_quad.astype(np.float32)
    )
    return cv2.warpPerspective(
        src,
        M,
        (CANVAS_W, CANVAS_H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(bg, bg, bg) if src.ndim == 3 else bg,
    )


def aruco_scene(
    image_quad_px: np.ndarray,
    marker_length_mm: float = 40.0,
    marker_id: int = 7,
    dictionary: str = "DICT_4X4_50",
) -> tuple[np.ndarray, np.ndarray]:
    """Render an ArUco marker so its outer black square lands on image_quad_px.

    Returns (frame, H_board_to_image). Board coordinates are the marker's own
    frame: (0,0) at its top-left corner, millimetres, x right and y down.
    """
    side_px = 400
    pad = 120
    marker = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary)),
        marker_id,
        side_px,
    )
    # White quiet zone. The detector needs it and a real phone screen has one.
    src = np.full((side_px + 2 * pad, side_px + 2 * pad), 255, np.uint8)
    src[pad : pad + side_px, pad : pad + side_px] = marker
    src_quad = quad(
        pad + side_px / 2, pad + side_px / 2, side_px, side_px
    )  # the marker's own outline inside src

    frame = _render(src, src_quad, image_quad_px, bg=255)

    L = marker_length_mm
    board = np.array([[0, 0], [L, 0], [L, L], [0, L]], dtype=np.float64)
    return frame, board_to_image(board, image_quad_px)


def card_scene(
    image_quad_px: np.ndarray,
    width_mm: float = 85.60,
    height_mm: float = 53.98,
    bow_mm: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Render a bright ID-1 card on a dark background onto image_quad_px.

    bow_mm bends the card's long edges outward by that many millimetres at their
    midpoint before warping. A real card that isn't flat, or a lens with barrel
    distortion, looks like this — and it's the thing the outline residual is
    supposed to notice.
    """
    scale = 8.0  # source pixels per mm, generous so the render is clean
    cw, ch = int(round(width_mm * scale)), int(round(height_mm * scale))
    pad = 80
    src = np.full((ch + 2 * pad, cw + 2 * pad, 3), 35, np.uint8)

    if bow_mm == 0.0:
        cv2.rectangle(src, (pad, pad), (pad + cw, pad + ch), (235, 235, 235), -1)
    else:
        b = bow_mm * scale
        # A closed curve with bulging top and bottom edges instead of straight ones.
        top = [(pad + t * cw, pad - b * np.sin(np.pi * t)) for t in np.linspace(0, 1, 60)]
        bottom = [
            (pad + (1 - t) * cw, pad + ch + b * np.sin(np.pi * t))
            for t in np.linspace(0, 1, 60)
        ]
        poly = np.array(top + bottom, dtype=np.int32)
        cv2.fillPoly(src, [poly], (235, 235, 235))

    src_quad = quad(pad + cw / 2, pad + ch / 2, cw, ch)
    frame = _render(src, src_quad, image_quad_px, bg=35)

    board = np.array(
        [[0, 0], [width_mm, 0], [width_mm, height_mm], [0, height_mm]], dtype=np.float64
    )
    return frame, board_to_image(board, image_quad_px)


def blank_scene() -> np.ndarray:
    """A frame with nothing calibratable in it. Some texture, so the test isn't
    passing only because the image is uniform."""
    rng = np.random.default_rng(0)
    frame = rng.integers(60, 90, size=(CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    cv2.circle(frame, (700, 450), 180, (150, 150, 150), -1)
    return frame
