"""The web app.

    python -m api.server
    # or: uvicorn api.server:app

Serves one HTML file, one WebSocket, and the REST endpoints in routes.py. The
pipeline it runs is the same Pipeline the batch CLI runs — there is no separate
"live" code path, so anything the browser shows also shows up in a CSV.
"""

from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from api.routes import router
from api.ws import Hub, LiveSession, LiveSettings
from pipeline.config import REPO_ROOT, load_config, resolve_path
from pipeline.pipeline import Pipeline
from store import db as store_db

WEB_INDEX = REPO_ROOT / "web" / "index.html"


def build_app(config_path: str | None = None, autostart: bool = True) -> FastAPI:
    cfg = load_config(config_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if autostart and app.state.live.settings.source:
            src = Path(app.state.live.settings.source)
            # Only autostart if the source is actually there. A server that dies
            # on boot because a demo folder is missing is a bad first impression
            # for someone who just cloned this.
            if src.exists() or str(src).isdigit() or str(src) == "webcam":
                await app.state.live.start()
        yield
        await app.state.live.stop()
        app.state.conn.close()

    app = FastAPI(title="fingerling", lifespan=lifespan)

    conn = store_db.connect(resolve_path(cfg.get("paths", {}).get("db", "data/fingerling.db")))
    live_cfg = cfg.get("live", {})

    app.state.cfg = cfg
    app.state.conn = conn
    app.state.hub = Hub()
    app.state.pipeline = Pipeline(cfg, conn=conn, save_frames=bool(live_cfg.get("save_frames", False)))
    app.state.live = LiveSession(
        app.state.pipeline,
        app.state.hub,
        LiveSettings(
            source=str(live_cfg.get("source", "data/sample")),
            fps=float(live_cfg.get("fps", 6.0)),
            loop=bool(live_cfg.get("loop", True)),
        ),
    )

    app.include_router(router)

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        if not WEB_INDEX.exists():
            return HTMLResponse("<h1>web/index.html is missing</h1>", status_code=500)
        return HTMLResponse(WEB_INDEX.read_text())

    @app.get("/healthz")
    def healthz() -> JSONResponse:
        return JSONResponse({"ok": True, "detector": app.state.pipeline.detector.name})

    @app.websocket("/ws")
    async def stream(ws: WebSocket) -> None:
        await app.state.hub.join(ws)
        try:
            # Send the current status immediately so a browser that connects
            # while the source is stopped shows the right thing instead of a
            # blank panel waiting for a frame that isn't coming.
            await ws.send_json({"type": "status", "status": app.state.live.status()})
            while True:
                await ws.receive_text()  # nothing to receive; this parks the socket
        except WebSocketDisconnect:
            pass
        finally:
            app.state.hub.leave(ws)

    return app


app = build_app()


def main() -> int:
    import uvicorn

    ap = argparse.ArgumentParser(prog="python -m api.server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    uvicorn.run(build_app(args.config), host=args.host, port=args.port, ws="auto")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
