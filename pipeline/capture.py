"""Getting frames in, from the three places they come from.

A note on shape: the three `from_*` functions are ordinary functions that RETURN
a generator, rather than being generators themselves. That's deliberate and a
test caught it. As generators, "no such file" wasn't raised until the first
`next()`, which is inside the pipeline's loop and well past the CLI's error
handling. A source that can't be opened should fail at the moment you try to
open it.

Webcam, video file, directory of images. One iterator interface over all three so
nothing downstream knows or cares which it's reading. The batch CLI and the live
server run the identical pipeline.

Frames get downscaled to `capture.max_long_edge_px` before anything sees them.
That's not cosmetic: detection and calibration cost scale with pixel count, so
without it the timing numbers from a 4K video and a 720p webcam aren't comparable
and the whole per-stage latency story falls apart.

The scaling is recorded on the frame but NOT undone anywhere. Every coordinate in
this system is in downscaled-frame pixels, including the ones written to the
database. If you ever need to point at the original image, multiply by `scale`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class Frame:
    image: np.ndarray
    index: int
    source: str  # a human-readable description, stored on the record
    path: str | None = None  # set for image directories, None for video/webcam
    scale: float = 1.0  # original pixels per current pixel; 1.0 means no resize


def _fit(image: np.ndarray, max_long_edge: int | None) -> tuple[np.ndarray, float]:
    if not max_long_edge:
        return image, 1.0
    h, w = image.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_long_edge:
        return image, 1.0
    f = max_long_edge / long_edge
    out = cv2.resize(image, (int(round(w * f)), int(round(h * f))), interpolation=cv2.INTER_AREA)
    return out, 1.0 / f


class CaptureError(RuntimeError):
    pass


def from_webcam(index: int = 0, max_long_edge: int | None = 1280, limit: int | None = None) -> Iterator[Frame]:
    """Frames off a camera, forever, until the camera stops or `limit` is hit.

    No reconnection logic. If the camera drops, this stops, because a grading station
    that silently reconnected to a different device would be worse than one that
    halted and said so.
    """
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise CaptureError(f"could not open camera index {index}")
    return _webcam_frames(cap, index, max_long_edge, limit)


def _webcam_frames(cap, index, max_long_edge, limit) -> Iterator[Frame]:
    try:
        i = 0
        while limit is None or i < limit:
            ok, image = cap.read()
            if not ok:
                break
            img, scale = _fit(image, max_long_edge)
            yield Frame(img, i, f"webcam:{index}", None, scale)
            i += 1
    finally:
        cap.release()


def from_video(
    path: str | Path,
    max_long_edge: int | None = 1280,
    stride: int = 1,
    limit: int | None = None,
) -> Iterator[Frame]:
    """Frames from a video file. `stride` takes every Nth frame.

    Stride exists because a 30 fps video of one fish on a tray is 30 nearly
    identical frames per second, and grading every one of them fills the database
    with duplicates of the same fish. It is NOT fish tracking. This project has
    no notion of the same fish across frames, and the scope doesn't ask
    for one. Each frame is an independent grading event.
    """
    p = Path(path)
    if not p.exists():
        raise CaptureError(f"no such video file: {p}")
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise CaptureError(f"could not open video {p}")
    return _video_frames(cap, p, max_long_edge, stride, limit)


def _video_frames(cap, p, max_long_edge, stride, limit) -> Iterator[Frame]:
    try:
        raw = 0
        emitted = 0
        while limit is None or emitted < limit:
            ok, image = cap.read()
            if not ok:
                break
            if raw % max(stride, 1) == 0:
                img, scale = _fit(image, max_long_edge)
                yield Frame(img, emitted, f"video:{p.name}", str(p), scale)
                emitted += 1
            raw += 1
    finally:
        cap.release()


def from_images(
    directory: str | Path, max_long_edge: int | None = 1280, limit: int | None = None
) -> Iterator[Frame]:
    """Every image in a directory, sorted by filename so runs are reproducible."""
    d = Path(directory)
    if not d.is_dir():
        raise CaptureError(f"not a directory: {d}")
    files = sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise CaptureError(f"no images in {d}")
    return _image_frames(files, d, max_long_edge, limit)


def _image_frames(files, d, max_long_edge, limit) -> Iterator[Frame]:
    for i, p in enumerate(files):
        if limit is not None and i >= limit:
            break
        image = cv2.imread(str(p))
        if image is None:
            continue  # unreadable file; skip rather than kill the batch
        img, scale = _fit(image, max_long_edge)
        yield Frame(img, i, f"images:{d.name}", str(p), scale)


def open_source(
    spec: str | Path, cfg: dict[str, Any], stride: int = 1, limit: int | None = None
) -> Iterator[Frame]:
    """Work out what `spec` is and open it.

    A bare integer, or "webcam", means a camera. A directory means images. Any
    other path means a video file. That covers the three sources the spec asks
    for without a --source-type flag nobody would remember.
    """
    max_edge = cfg.get("capture", {}).get("max_long_edge_px", 1280)
    s = str(spec)

    if s == "webcam":
        return from_webcam(int(cfg.get("capture", {}).get("camera_index", 0)), max_edge, limit)
    if s.isdigit() and not Path(s).exists():
        return from_webcam(int(s), max_edge, limit)
    p = Path(s)
    if p.is_dir():
        return from_images(p, max_edge, limit)
    return from_video(p, max_edge, stride, limit)
