"""Video in, CSV out. No UI, no server.

    python -m cli.batch input.mp4 --out results.csv

This is the mode that matters for anything except demoing: point it at a folder
of images or a video and get one row per graded fish with the full per-stage
timing breakdown. It writes to the same SQLite database the web app reads, so a
batch run and a live session end up in one place.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from pipeline.capture import CaptureError, open_source
from pipeline.config import load_config, resolve_path
from pipeline.pipeline import Pipeline
from store import db as store_db

# What lands in the CSV. Not every DB column — landmarks_json is thousands of
# characters and belongs in the database, not in a spreadsheet. The timing
# columns are split out of latency_ms_json into real columns because the whole
# point of a CSV is that someone can pivot it.
CSV_COLUMNS = (
    "id", "timestamp", "source", "frame_index", "decision", "trust_score",
    "landmark_confidence", "plausibility_score", "failed_constraints",
    "calibrated", "calibration_target", "calibration_residual",
    "calibration_obliquity", "calibration_reliable",
    "fork_length_mm", "fork_length_mm_sigma", "total_length_mm",
    "body_depth_mm", "body_depth_mm_sigma", "depth_ratio", "depth_ratio_sigma",
    "peduncle_depth_mm", "peduncle_depth_mm_sigma",
    "curvature_index", "curvature_index_sigma",
    "weight_g", "condition_factor", "detector_source", "frame_path",
    "latency_detect_ms", "latency_calibrate_ms", "latency_measure_ms",
    "latency_trust_ms", "latency_decide_ms", "latency_total_ms",
)


def _csv_row(record) -> dict:
    row = record.to_row()
    lat = record.latency_ms
    out = {
        "id": record.id,
        "frame_index": record.frame_index,
        "failed_constraints": "|".join(record.trust.failed_names),
    }
    for c in CSV_COLUMNS:
        if c in out:
            continue
        if c.startswith("latency_"):
            stage = c.removeprefix("latency_").removesuffix("_ms")
            out[c] = round(lat.get(stage, float("nan")), 3) if stage in lat else None
        else:
            out[c] = row.get(c)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m cli.batch",
        description="Grade every frame of a video, image directory, or webcam and write a CSV.",
    )
    ap.add_argument(
        "source",
        help="video file, directory of images, a camera index, or the word 'webcam'",
    )
    ap.add_argument("--out", default="results.csv", help="CSV path (default: results.csv)")
    ap.add_argument("--config", default=None, help="config.yaml path")
    ap.add_argument("--stride", type=int, default=1, help="take every Nth video frame")
    ap.add_argument("--limit", type=int, default=None, help="stop after N frames")
    ap.add_argument("--db", default=None, help="override the database path")
    ap.add_argument("--no-store", action="store_true", help="CSV only, do not write the database")
    ap.add_argument("--save-frames", action="store_true", help="save each graded frame to disk")
    ap.add_argument("--quiet", action="store_true", help="no per-frame progress")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)

    conn = None
    if not args.no_store:
        db_path = args.db or cfg.get("paths", {}).get("db", "data/fingerling.db")
        conn = store_db.connect(resolve_path(db_path))

    pipe = Pipeline(cfg, conn=conn, save_frames=args.save_frames)

    try:
        frames = open_source(args.source, cfg, stride=args.stride, limit=args.limit)
    except CaptureError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    graded = 0
    seen = 0
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for record in pipe.run(frames, store=conn is not None):
            seen += 1
            if not record.landmarks.detected:
                continue
            writer.writerow(_csv_row(record))
            graded += 1
            if not args.quiet and graded % 25 == 0:
                print(f"  {graded} graded / {seen} frames", file=sys.stderr)

    print(f"{graded} fish graded from {seen} frames -> {out_path}")

    if conn is not None:
        stats = store_db.latency_percentiles(conn)
        if stats:
            print("\nper-stage latency over everything in the database, ms:")
            print(f"  {'stage':<12}{'n':>7}{'p50':>9}{'p95':>9}{'mean':>9}")
            order = ["detect", "calibrate", "measure", "trust", "decide", "save_frame", "total"]
            for stage in order + [s for s in stats if s not in order]:
                if stage not in stats:
                    continue
                s = stats[stage]
                print(
                    f"  {stage:<12}{s['n']:>7}{s['p50']:>9.2f}{s['p95']:>9.2f}{s['mean']:>9.2f}"
                )
        fired = store_db.constraint_counts(conn)
        if fired:
            total = store_db.count(conn)
            print(f"\nplausibility constraints that failed, over {total} rows:")
            for name, n in fired.items():
                print(f"  {name:38}{n:>6}")

        conn.close()

    if graded == 0:
        print(
            "nothing was graded. If the detector backend is 'stub' it only "
            "produces a fish when its mode isn't 'no_detection'; if it's 'yolo', "
            "check the weights path.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
