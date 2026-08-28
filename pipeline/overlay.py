"""Drawing the landmarks onto a frame.

Lives in pipeline/ and not api/ because pipeline/ is not allowed to import api/,
and the batch CLI wants to dump annotated frames too. It's an extra module that
isn't in CLAUDE.md's architecture listing — logged in docs/DECISIONS.md.

Colour carries the one thing an operator needs at a glance: the decision. Green
passes, red culls, amber goes to a human. Landmark dots are shaded by their own
confidence, so a point the model is unsure about looks unsure.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from pipeline.landmarks import MIDLINE_CHAIN, Landmarks, midline_points

DECISION_BGR = {
    "PASS": (90, 200, 90),
    "CULL": (60, 60, 220),
    "REVIEW": (40, 180, 235),
}

# Body outline drawn as a path through the landmarks, so the overlay reads as a
# fish rather than a cloud of dots. Purely cosmetic — nothing measures this.
OUTLINE = (
    "snout_tip", "dorsal_origin", "dorsal_apex", "dorsal_insertion",
    "peduncle_dorsal", "caudal_tip", "peduncle_ventral", "ventral_margin",
)


def draw(
    frame: np.ndarray,
    lm: Landmarks,
    decision: str | None = None,
    calib: Any = None,
    label_lines: list[str] | None = None,
) -> np.ndarray:
    """Return an annotated copy. Never mutates the frame it was given — the
    pipeline may still want the clean one to save."""
    out = frame.copy()
    colour = DECISION_BGR.get(decision or "", (200, 200, 200))

    if calib is not None and getattr(calib, "corners_px", None) is not None:
        q = np.asarray(calib.corners_px, dtype=np.int32).reshape(-1, 1, 2)
        cal_colour = (200, 200, 90) if getattr(calib, "reliable", False) else (90, 90, 200)
        cv2.polylines(out, [q], True, cal_colour, 2)
        cv2.putText(
            out, f"cal:{getattr(calib, 'target', '?')}",
            tuple(np.asarray(calib.corners_px, dtype=int)[0] + np.array([0, -8])),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, cal_colour, 1, cv2.LINE_AA,
        )

    if lm.detected:
        pts = [lm.point(n) for n in OUTLINE if lm.has(n)]
        if len(pts) >= 3:
            cv2.polylines(
                out, [np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)],
                True, colour, 2, cv2.LINE_AA,
            )

        mid = midline_points(lm)
        if mid is not None:
            cv2.polylines(
                out, [mid.astype(np.int32).reshape(-1, 1, 2)],
                False, (230, 230, 230), 1, cv2.LINE_AA,
            )

        for i, name in enumerate(lm.schema):
            if not lm.present[i]:
                continue
            x, y = lm.points[i].astype(int)
            c = float(lm.confidence[i])
            # Dim the dot in proportion to confidence. A low-confidence landmark
            # that looks identical to a high-confidence one is a UI that lies.
            shade = int(60 + 195 * max(0.0, min(1.0, c)))
            cv2.circle(out, (x, y), 4, (shade, shade, shade), -1, cv2.LINE_AA)
            cv2.circle(out, (x, y), 4, colour, 1, cv2.LINE_AA)

    y = 24
    for line in [decision or "NO DETECTION"] + (label_lines or []):
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 1, cv2.LINE_AA)
        y += 22
    return out


def summary_lines(record: Any) -> list[str]:
    """Two or three lines of the numbers that matter, for the overlay caption."""
    m = record.measures
    lines: list[str] = []
    if m.fork_length_mm is not None:
        lines.append(f"fork {m.fork_length_mm.value:.1f} +/- {m.fork_length_mm.sigma:.1f} mm")
    elif m.fork_length_px is not None:
        lines.append(f"fork {m.fork_length_px.value:.0f} px (uncalibrated)")
    if m.curvature_index is not None:
        lines.append(f"curvature {m.curvature_index.value:.3f}")
    lines.append(f"trust {record.trust.trust_score:.2f}")
    if record.trust.failed_names:
        lines.append("failed: " + ", ".join(record.trust.failed_names[:2]))
    return lines
