"""API tests, against a real server over a real socket.

FastAPI's TestClient would be lighter, but it doesn't exercise the WebSocket path
the way uvicorn does and the WS transport is exactly the bit that was uncertain
(uvicorn ships no WebSocket implementation of its own). So these run a real
uvicorn on a real port.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path

import pytest

uvicorn = pytest.importorskip("uvicorn")
websockets = pytest.importorskip("websockets")
import urllib.error
import urllib.request

from api.server import build_app
from pipeline.config import load_config


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    import cv2

    from cli.make_sample import frames as sample_frames

    d = tmp_path_factory.mktemp("clip")
    for i, frame in enumerate(sample_frames(8, seed=2)):
        cv2.imwrite(str(d / f"f{i:03d}.png"), frame)
    return d


@pytest.fixture(scope="module")
def server(tmp_path_factory, sample):
    """A real uvicorn on a real port, grading the sample clip on a loop."""
    cfg_path = tmp_path_factory.mktemp("cfg") / "config.yaml"
    cfg = load_config()
    cfg["paths"] = dict(cfg["paths"], db=str(tmp_path_factory.mktemp("db") / "t.db"))
    cfg["live"] = {"source": str(sample), "fps": 30.0, "loop": True, "save_frames": False}
    # A mode that routes to REVIEW, so the review queue has something in it and
    # the correction endpoint has something to correct.
    cfg["detector"] = dict(cfg["detector"], stub_mode="implausible")
    import yaml

    cfg_path.write_text(yaml.safe_dump(cfg))

    port = free_port()
    app = build_app(str(cfg_path))
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", ws="auto")
    srv = uvicorn.Server(config)
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()

    base = f"http://127.0.0.1:{port}"
    for _ in range(200):
        try:
            urllib.request.urlopen(base + "/healthz", timeout=0.5).read()
            break
        except Exception:  # noqa: BLE001
            time.sleep(0.05)
    else:
        raise RuntimeError("server never came up")

    # Let the live loop grade a few frames before the tests look at the tables.
    deadline = time.time() + 10
    while time.time() < deadline:
        if get(base, "/api/stats")["counts"]["total"] >= 4:
            break
        time.sleep(0.1)

    yield base

    srv.should_exit = True
    thread.join(timeout=10)


def get(base: str, path: str):
    with urllib.request.urlopen(base + path, timeout=5) as r:
        return json.loads(r.read())


def post(base: str, path: str, payload: dict | None = None):
    data = json.dumps(payload or {}).encode()
    req = urllib.request.Request(
        base + path, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


# ---------------------------------------------------------------------------


def test_the_page_is_served_and_is_one_file(server):
    with urllib.request.urlopen(server + "/", timeout=5) as r:
        html = r.read().decode()
    assert "<title>fingerling</title>" in html
    # No build step, no framework. If either appears, the scope slipped.
    assert "<script src=" not in html
    assert "react" not in html.lower()


def test_the_live_loop_is_grading_and_storing(server):
    counts = get(server, "/api/stats")["counts"]
    assert counts["total"] >= 4


def test_records_come_back_with_decoded_json_columns(server):
    recs = get(server, "/api/records?limit=5")["records"]
    assert recs
    r = recs[0]
    assert isinstance(r["failed_constraints"], list)
    assert isinstance(r["latency_ms"], dict)
    assert "total" in r["latency_ms"]
    assert isinstance(r["landmarks"], dict)


def test_stats_expose_per_stage_latency(server):
    lat = get(server, "/api/stats")["latency_ms"]
    for stage in ("detect", "calibrate", "measure", "trust", "decide", "total"):
        assert stage in lat
        assert lat[stage]["p95"] >= lat[stage]["p50"]


def test_the_review_queue_lists_the_implausible_fish_with_named_constraints(server):
    recs = get(server, "/api/records?needs_review=true&limit=5")["records"]
    assert recs, "the implausible stub should have filled the review queue"
    assert "peduncle_points_not_swapped" in recs[0]["failed_constraints"]


def test_correcting_a_fish_records_the_human_and_removes_it_from_the_queue(server):
    before = get(server, "/api/records?needs_review=true&limit=50")["records"]
    target = before[0]["id"]

    updated = post(server, f"/api/records/{target}/correct", {"decision": "PASS", "note": "fine"})
    assert updated["human_corrected"] == 1
    assert updated["human_decision"] == "PASS"
    assert updated["human_note"] == "fine"
    # The machine's own verdict must survive — it's half the training pair.
    assert updated["decision"] == "REVIEW"

    still = [r["id"] for r in get(server, "/api/records?needs_review=true&limit=50")["records"]]
    assert target not in still


def test_a_nonsense_decision_is_rejected(server):
    recs = get(server, "/api/records?limit=1")["records"]
    with pytest.raises(urllib.error.HTTPError) as exc:
        post(server, f"/api/records/{recs[0]['id']}/correct", {"decision": "MAYBE"})
    assert exc.value.code == 400


def test_correcting_a_fish_that_does_not_exist_is_a_404(server):
    with pytest.raises(urllib.error.HTTPError) as exc:
        post(server, "/api/records/999999/correct", {"decision": "PASS"})
    assert exc.value.code == 404


def test_export_is_a_csv_with_a_header_and_the_rows(server):
    with urllib.request.urlopen(server + "/api/export.csv", timeout=10) as r:
        assert r.headers["Content-Type"].startswith("text/csv")
        body = r.read().decode()
    lines = body.strip().splitlines()
    assert lines[0].startswith("id,timestamp,source,decision")
    assert len(lines) > 1


def test_start_and_stop_actually_change_the_live_state(server):
    assert post(server, "/api/live/stop")["running"] is False
    assert get(server, "/api/live")["running"] is False
    assert post(server, "/api/live/start")["running"] is True


def test_websocket_delivers_a_frame_and_a_record(server):
    """The transport question from DECISIONS.md, answered by actually connecting.

    If `websockets` weren't installed, uvicorn would refuse this handshake and
    this test would fail — which is the point of testing it over a real socket
    rather than through TestClient.
    """
    import asyncio

    async def go():
        url = server.replace("http://", "ws://") + "/ws"
        async with websockets.connect(url, open_timeout=10) as ws:
            deadline = asyncio.get_event_loop().time() + 20
            while asyncio.get_event_loop().time() < deadline:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=20))
                if msg["type"] == "frame":
                    return msg
        raise AssertionError("no frame arrived on the websocket")

    msg = asyncio.run(go())
    assert msg["image"].startswith("data:image/jpeg;base64,")
    assert msg["record"]["decision"] in ("PASS", "CULL", "REVIEW")
    assert "constraints" in msg["record"]
    assert msg["status"]["running"] is True
