"""WebSocket streaming: annotated frames and records out to the browser.

The transport question that was sitting open in docs/DECISIONS.md is settled.
`websockets` is now a dependency. uvicorn on its own has no WebSocket
implementation and `ws="auto"` resolves to none, so without it a WS connection is
simply refused. The alternatives were wsproto (same job, less widely used),
uvicorn[standard] (pulls in several more packages for things I don't need), or
dropping to server-sent events. One package for the thing the spec's design
actually asks for was the cheapest of those.

Design notes:

  Frames go out as base64 JPEG inside the JSON message. That is not how you'd
  build a video pipeline (it's roughly 33% bigger than the bytes and it costs a
  copy) but it keeps the frame and the record it belongs to in ONE message, so
  the overlay can never be a frame ahead of the numbers beside it. On a single
  station at single-digit fps that tradeoff is free, and the alternative
  (a binary channel plus correlation ids) is a lot of machinery for one operator
  looking at one tray.

  A slow client is dropped from the broadcast, not queued. A grading station's
  live view showing a frame from thirty seconds ago is worse than one that
  reconnects.
"""

from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass
from typing import Any, Iterator

import cv2

from pipeline import overlay
from pipeline.capture import CaptureError, Frame, open_source
from pipeline.pipeline import FishRecord, Pipeline


def encode_frame(image, quality: int = 72) -> str:
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


class Hub:
    """Every connected browser. Broadcast is best-effort, by design."""

    def __init__(self) -> None:
        self.clients: set[Any] = set()

    async def join(self, ws) -> None:
        await ws.accept()
        self.clients.add(ws)

    def leave(self, ws) -> None:
        self.clients.discard(ws)

    async def broadcast(self, message: dict[str, Any]) -> None:
        if not self.clients:
            return
        payload = json.dumps(message)
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_text(payload)
            except Exception:  # noqa: BLE001. A dead socket is not an error here
                dead.append(ws)
        for ws in dead:
            self.leave(ws)


@dataclass
class LiveSettings:
    source: str = "data/sample"
    fps: float = 6.0
    loop: bool = True


class LiveSession:
    """Runs the pipeline over a capture source and pushes results to the Hub.

    One asyncio task, not a thread. The pipeline is synchronous and CPU-bound, so
    each frame briefly blocks the event loop, but at a few frames a second on a
    single-station tool that's fine, and it keeps the whole thing to one place
    where state lives. If this ever needed to run at camera rate it would move to
    a thread with a queue, and that's a real change, not a tweak.
    """

    def __init__(self, pipeline: Pipeline, hub: Hub, settings: LiveSettings) -> None:
        self.pipeline = pipeline
        self.hub = hub
        self.settings = settings
        self.task: asyncio.Task | None = None
        self.running = False
        self.last_error: str | None = None
        self.frames_seen = 0
        self.fish_graded = 0

    def status(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "source": self.settings.source,
            "fps": self.settings.fps,
            "loop": self.settings.loop,
            "frames_seen": self.frames_seen,
            "fish_graded": self.fish_graded,
            "last_error": self.last_error,
            "detector": self.pipeline.detector.name,
            "clients": len(self.hub.clients),
        }

    async def start(self, settings: LiveSettings | None = None) -> dict[str, Any]:
        await self.stop()
        if settings is not None:
            self.settings = settings
        self.last_error = None
        self.running = True
        self.task = asyncio.create_task(self._loop())
        return self.status()

    async def stop(self) -> dict[str, Any]:
        self.running = False
        if self.task is not None:
            self.task.cancel()
            try:
                await self.task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self.task = None
        return self.status()

    def _open(self) -> Iterator[Frame]:
        return open_source(self.settings.source, self.pipeline.cfg)

    async def _loop(self) -> None:
        delay = 1.0 / max(self.settings.fps, 0.1)
        try:
            while self.running:
                try:
                    frames = self._open()
                except CaptureError as exc:
                    self.last_error = str(exc)
                    self.running = False
                    await self.hub.broadcast({"type": "error", "message": str(exc)})
                    return

                for frame in frames:
                    if not self.running:
                        return
                    record = self.pipeline.process(frame)
                    self.frames_seen += 1
                    if record.landmarks.detected and self.pipeline.conn is not None:
                        from store import db as _db

                        record.id = _db.insert(self.pipeline.conn, record.to_row())
                        self.fish_graded += 1
                    await self.hub.broadcast(self._message(frame, record))
                    await asyncio.sleep(delay)

                if not self.settings.loop:
                    self.running = False
                    await self.hub.broadcast({"type": "ended"})
                    return
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.running = False
            await self.hub.broadcast({"type": "error", "message": self.last_error})

    def _message(self, frame: Frame, record: FishRecord) -> dict[str, Any]:
        annotated = overlay.draw(
            frame.image,
            record.landmarks,
            record.decision.decision if record.landmarks.detected else None,
            record.calibration,
            overlay.summary_lines(record) if record.landmarks.detected else [],
        )
        return {
            "type": "frame",
            "image": encode_frame(annotated),
            "record": record.to_public(),
            "status": self.status(),
        }
