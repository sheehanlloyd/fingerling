"""The whole thing wired together, with a stopwatch on every stage.

capture -> detect -> calibrate -> measure -> trust -> decide -> record.

Timing is built in from the first commit rather than bolted on later, because
the latency story in Phase 3 depends on having measured it all along. Every
stage is timed separately with `perf_counter` and the breakdown is stored per
fish, so p50/p95 per stage is a query against real history and not a benchmark
someone ran once on a good day (`store.db.latency_percentiles`).

`pipeline/` imports nothing from `api/`. That's a rule in CLAUDE.md and it's the
reason this module is usable as a library: the batch CLI and the web server run
this exact code path, so anything the UI shows, the CSV also shows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator

import cv2
import numpy as np

from pipeline.calibrate import Calibration, CalibrationSettings, calibrate
from pipeline.capture import Frame
from pipeline.config import resolve_path
from pipeline.decide import Decision, DecideSettings, decide
from pipeline.detect import Detector, build_detector
from pipeline.landmarks import Landmarks
from pipeline.measure import MeasureSettings, MeasurementSet, measure_traits
from pipeline.trust import TrustResult, TrustSettings, score_trust


class _Stopwatch:
    """Accumulates per-stage milliseconds. Deliberately dumb — a context manager
    per stage, one dict out. Overhead is a couple of microseconds, which is well
    under the resolution anything here is measured at."""

    def __init__(self) -> None:
        self.ms: dict[str, float] = {}
        self._t0 = perf_counter()

    def stage(self, name: str) -> "_StageTimer":
        return _StageTimer(self, name)

    def finish(self) -> dict[str, float]:
        self.ms["total"] = (perf_counter() - self._t0) * 1000.0
        return self.ms


class _StageTimer:
    def __init__(self, sw: _Stopwatch, name: str) -> None:
        self.sw, self.name = sw, name

    def __enter__(self) -> "_StageTimer":
        self.t = perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.sw.ms[self.name] = (perf_counter() - self.t) * 1000.0


@dataclass
class FishRecord:
    """Everything the pipeline knows about one frame's fish."""

    timestamp: str
    source: str
    frame_path: str | None
    landmarks: Landmarks
    calibration: Calibration
    measures: MeasurementSet
    trust: TrustResult
    decision: Decision
    latency_ms: dict[str, float]
    frame_index: int = 0
    weight_g: float | None = None
    id: int | None = None

    def to_row(self) -> dict[str, Any]:
        """Flatten to the database's column names."""
        m, t, c = self.measures, self.trust, self.calibration

        def val(q):
            return None if q is None else float(q.value)

        def sig(q):
            return None if q is None else float(q.sigma)

        return {
            "timestamp": self.timestamp,
            "source": self.source,
            "frame_path": self.frame_path,
            "landmarks_json": json.dumps(self.landmarks.as_dict()),
            "calibrated": int(bool(c.calibrated)),
            # Only stored when it means something. An ArUco residual is ~0 by
            # construction and writing that into a column called "residual"
            # invites someone to read it as a quality figure later.
            "calibration_residual": (
                float(c.residual_px) if (c.residual_px is not None and c.residual_meaningful) else None
            ),
            "fork_length_mm": val(m.fork_length_mm),
            "body_depth_mm": val(m.body_depth_mm),
            "depth_ratio": val(m.depth_ratio),
            "peduncle_depth_mm": val(m.peduncle_depth_mm),
            "curvature_index": val(m.curvature_index),
            "weight_g": self.weight_g,
            "condition_factor": val(m.condition_factor),
            "landmark_confidence": float(t.landmark_confidence),
            "plausibility_score": float(t.plausibility_score),
            "failed_constraints_json": json.dumps(t.failed_names),
            "trust_score": float(t.trust_score),
            "decision": self.decision.decision,
            "human_corrected": 0,
            "human_decision": None,
            "latency_ms_json": json.dumps({k: round(v, 3) for k, v in self.latency_ms.items()}),
            "fork_length_mm_sigma": sig(m.fork_length_mm),
            "body_depth_mm_sigma": sig(m.body_depth_mm),
            "depth_ratio_sigma": sig(m.depth_ratio),
            "peduncle_depth_mm_sigma": sig(m.peduncle_depth_mm),
            "curvature_index_sigma": sig(m.curvature_index),
            "detector_source": self.landmarks.source,
            "calibration_target": c.target,
            "calibration_obliquity": (None if c.obliquity is None else float(c.obliquity)),
            "calibration_reliable": int(bool(c.reliable)),
            "total_length_mm": val(m.total_length_mm),
            "decision_reasons_json": json.dumps(self.decision.reasons),
            "corrected_at": None,
            "human_note": None,
        }

    def to_public(self) -> dict[str, Any]:
        """What the websocket and REST layers send. Same numbers as the DB row,
        plus the constraint detail the review queue needs to explain itself."""
        row = self.to_row()
        row["id"] = self.id
        row["frame_index"] = self.frame_index
        row["constraints"] = [c.as_dict() for c in self.trust.constraints]
        row["landmarks"] = self.landmarks.as_dict()
        row["unavailable"] = dict(self.measures.unavailable)
        row["calibration_reason"] = self.calibration.reason
        return row


class Pipeline:
    """Holds the settings and the detector so they're built once, not per frame.

    Building a detector per frame would dominate the timing numbers for the real
    model — loading weights takes far longer than inference — and would make the
    per-stage breakdown a lie.
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        conn: Any | None = None,
        save_frames: bool = False,
        detector: Detector | None = None,
    ) -> None:
        self.cfg = cfg
        self.conn = conn
        self.save_frames = save_frames
        self.detector = detector if detector is not None else build_detector(cfg)
        self.calibration_settings = CalibrationSettings.from_config(cfg)
        self.measure_settings = MeasureSettings.from_config(cfg)
        self.trust_settings = TrustSettings.from_config(cfg)
        self.decide_settings = DecideSettings.from_config(cfg)
        self.frames_dir = resolve_path(cfg.get("paths", {}).get("frames", "data/frames"))

    # -- the one method that matters -------------------------------------

    def process(self, frame: Frame, weight_g: float | None = None) -> FishRecord:
        sw = _Stopwatch()

        with sw.stage("detect"):
            lm = self.detector.detect(frame.image)

        with sw.stage("calibrate"):
            calib = calibrate(frame.image, self.calibration_settings)

        with sw.stage("measure"):
            measures = measure_traits(lm, calib, self.measure_settings, weight_g)

        with sw.stage("trust"):
            trust = score_trust(lm, measures, self.trust_settings)

        with sw.stage("decide"):
            decision = decide(measures, trust, self.decide_settings)

        frame_path = frame.path
        if self.save_frames and lm.detected:
            with sw.stage("save_frame"):
                frame_path = self._save_frame(frame)

        return FishRecord(
            timestamp=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            source=frame.source,
            frame_path=frame_path,
            landmarks=lm,
            calibration=calib,
            measures=measures,
            trust=trust,
            decision=decision,
            latency_ms=sw.finish(),
            frame_index=frame.index,
            weight_g=weight_g,
        )

    def _save_frame(self, frame: Frame) -> str:
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        name = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}_{frame.index:06d}.jpg"
        out = self.frames_dir / name
        cv2.imwrite(str(out), frame.image, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        return str(out)

    # -- convenience over a whole source ---------------------------------

    def run(
        self,
        frames: Iterator[Frame],
        store: bool = True,
        skip_undetected: bool = True,
    ) -> Iterator[FishRecord]:
        """Grade a whole source, yielding records as they're produced.

        `skip_undetected` is on by default: a frame with no fish in it is not a
        graded fish and writing a row for it would put thousands of empty records
        between the ones that matter. The count of skipped frames is not lost —
        the caller sees every frame go past and can count them.
        """
        from store import db as _db

        for frame in frames:
            record = self.process(frame)
            if skip_undetected and not record.landmarks.detected:
                yield record
                continue
            if store and self.conn is not None:
                record.id = _db.insert(self.conn, record.to_row())
            yield record
