"""Render a synthetic clip so the demo in the README works on a fresh clone.

    python -m cli.make_sample --out data/sample

There is no dataset in this repo and no fish on my desk, so "clone to running
demo" needs something to point the pipeline at. This writes frames containing a
credit-card-shaped calibration target that drifts and rotates slightly, which is
enough for the calibration stage to do real work.

Be clear about what this is NOT: there is no fish in these frames. The stub
detector doesn't look at the image at all. It draws a made-up fish wherever it
likes. So a batch run over this clip exercises capture, calibration, measurement,
trust, routing, storage and timing end to end with REAL calibration geometry and
a FAKE animal. Every millimetre it reports is a millimetre of the card's plane,
correctly computed, applied to a fish that isn't there.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from tests import synthetic as syn


def frames(n: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    for i in range(n):
        t = i / max(n - 1, 1)
        # A card wandering slowly across the frame with a little rotation and a
        # little perspective, the way one would if it were sitting on a tray
        # under a hand-held camera.
        q = syn.quad(
            520 + 120 * np.sin(2 * np.pi * t),
            470 + 60 * np.cos(2 * np.pi * t),
            420,
            420 * (53.98 / 85.60),
        )
        q = syn.rotate(q, 8.0 * np.sin(2 * np.pi * t + 0.7))
        q = syn.foreshorten(q, 1.0 - 0.06 * (0.5 + 0.5 * np.sin(2 * np.pi * t)))
        frame, _ = syn.card_scene(q)
        # A little noise, so nothing downstream is passing because the image is
        # unnaturally clean.
        frame = np.clip(
            frame.astype(np.int16) + rng.normal(0, 3, frame.shape).astype(np.int16), 0, 255
        ).astype(np.uint8)
        yield frame


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m cli.make_sample")
    ap.add_argument("--out", default="data/sample", help="output directory")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--video", default="data/sample.mp4", help="also write an mp4 here")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    writer = None
    if args.video:
        Path(args.video).parent.mkdir(parents=True, exist_ok=True)

    n = 0
    for i, frame in enumerate(frames(args.frames)):
        cv2.imwrite(str(out / f"frame_{i:04d}.png"), frame)
        if args.video:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(
                    args.video, cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (w, h)
                )
            writer.write(frame)
        n += 1
    if writer is not None:
        writer.release()

    print(f"wrote {n} frames to {out}" + (f" and {args.video}" if args.video else ""))
    print("no fish in these frames, see the docstring in cli/make_sample.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
