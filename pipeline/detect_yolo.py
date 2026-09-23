"""The real detector: a YOLO-pose model behind the same contract as the stub.

Imported lazily by pipeline/detect.py so the app still runs on a machine with no
torch and no ultralytics installed.

STATUS, read this before trusting anything here. As of the session that wrote it,
this adapter has been exercised against a real ultralytics pose model and a real
`Results` object, so the parsing is not guesswork. It has NOT been run against a
model trained on fish, because the dataset download is blocked on a Roboflow API
key, see docs/DECISIONS.md. So: the plumbing is tested, the model is not
trained, and `detector.backend` in config.yaml still defaults to `stub`.

The one piece of real judgement in here is the keypoint remap. A model trained on
someone else's annotations emits keypoints in THEIR order, which almost certainly
isn't pipeline/landmarks.py's SCHEMA order. Rather than reordering the dataset or
renumbering the schema, `detector.keypoint_order` in config lists the model's
names in the model's order and this class builds the permutation. A name in the
model that the schema doesn't have is dropped; a name in the schema the model
doesn't produce is marked absent, and every trait needing it goes null rather
than being computed off the wrong point.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from pipeline.landmarks import INDEX, N_LANDMARKS, SCHEMA, Landmarks


class YoloPoseDetector:
    """Ultralytics YOLO-pose, adapted to the Landmarks contract."""

    name = "yolo"

    def __init__(
        self,
        weights: str | Path | None,
        conf: float = 0.25,
        device: str = "mps",
        keypoint_order: list[str] | None = None,
        kpt_conf_min: float = 0.20,
    ) -> None:
        if not weights:
            raise ValueError("detector.weights is not set in config.yaml")
        p = Path(weights)
        if not p.exists():
            raise FileNotFoundError(f"no weights at {p}")

        from ultralytics import YOLO  # heavy; only imported when actually used

        self.model = YOLO(str(p))
        self.weights = str(p)
        self.conf = conf
        self.device = device
        self.kpt_conf_min = kpt_conf_min
        self.keypoint_order = list(keypoint_order) if keypoint_order else None
        self._perm = self._build_permutation(self.keypoint_order)

    @staticmethod
    def _build_permutation(order: list[str] | None) -> list[int] | None:
        """Map model keypoint index -> schema index, or None if already aligned."""
        if not order:
            return None
        perm: list[int] = []
        for name in order:
            perm.append(INDEX.get(name, -1))  # -1 means "the schema doesn't want this"
        return perm

    def detect(self, frame: np.ndarray) -> Landmarks:
        results = self.model.predict(
            frame, conf=self.conf, device=self.device, verbose=False
        )
        if not results:
            return Landmarks.empty(source=f"yolo:{Path(self.weights).name}")
        r = results[0]

        boxes = getattr(r, "boxes", None)
        kps = getattr(r, "keypoints", None)
        if boxes is None or kps is None or len(boxes) == 0:
            return Landmarks.empty(source=f"yolo:{Path(self.weights).name}")

        # One fish per frame. The scope is a single-station grader with
        # one fish under the camera; picking the most confident detection is the
        # honest version of that, and two overlapping fish is a documented
        # failure mode rather than something this silently averages.
        scores = boxes.conf.detach().cpu().numpy().reshape(-1)
        best = int(np.argmax(scores))

        xy = kps.xy.detach().cpu().numpy()[best]  # (K, 2)
        if getattr(kps, "conf", None) is not None:
            kconf = kps.conf.detach().cpu().numpy()[best]  # (K,)
        else:
            # A model exported without per-keypoint confidence gives us nothing to
            # threshold on. Treating that as "fully confident" would silently
            # inflate every trust score, so it's flagged instead.
            kconf = np.full(len(xy), float("nan"))

        points = np.zeros((N_LANDMARKS, 2), dtype=np.float64)
        confidence = np.zeros(N_LANDMARKS, dtype=np.float64)
        present = np.zeros(N_LANDMARKS, dtype=bool)

        for model_idx in range(len(xy)):
            schema_idx = (
                self._perm[model_idx]
                if self._perm is not None and model_idx < len(self._perm)
                else (model_idx if model_idx < N_LANDMARKS else -1)
            )
            if schema_idx < 0:
                continue
            c = float(kconf[model_idx]) if np.isfinite(kconf[model_idx]) else 1.0
            x, y = float(xy[model_idx][0]), float(xy[model_idx][1])
            # Ultralytics writes (0, 0) for a keypoint it did not place, which is
            # a real coordinate in image space, the top-left corner. Reading one
            # as a landmark puts a snout in the corner of the frame with full
            # confidence, so it's treated as absent.
            if c < self.kpt_conf_min or (x == 0.0 and y == 0.0):
                continue
            points[schema_idx] = (x, y)
            confidence[schema_idx] = c
            present[schema_idx] = True

        x1, y1, x2, y2 = boxes.xyxy.detach().cpu().numpy()[best].tolist()
        return Landmarks(
            detected=bool(present.any()),
            points=points,
            confidence=confidence,
            present=present,
            source=f"yolo:{Path(self.weights).name}",
            box_px=(x1, y1, x2, y2),
            detection_confidence=float(scores[best]),
        )
