"""Dataset plumbing for Phase 2: the Roboflow pull, and a synthetic stand-in.

Two things live here.

`fetch_roboflow` is the real path. It is written and it does not work tonight,
because it needs a ROBOFLOW_API_KEY and there isn't one in the environment.
Creating an account to get one is off the table, so this function exists ready to
run the moment a key shows up. Nothing about it is speculative — the endpoints
are documented and the code returns a clear error rather than a partial download.

`make_synthetic` writes a YOLO-pose dataset of geometric fish. It exists for one
narrow purpose: proving the TRAINING HARNESS runs end to end on this machine
(MPS, amp=False, the dataset YAML, the keypoint shape, the export) before the
real data arrives. A model trained on it has learned to find a polygon, and its
metrics say nothing whatsoever about fish. Do not quote them. Do not compare
them to anything.

FLIP AUGMENTATION IS OFF, and this is the one thing here worth remembering.
Ultralytics flips images horizontally by default and permutes keypoints through
`flip_idx` to match. That works for people, where a flip swaps left and right
wrists. It does NOT work for a fish photographed from the side: flipping the
image maps the snout onto the tail, and there is no permutation of these twelve
landmarks that expresses "the head is now where the tail was". So flip_idx is the
identity and fliplr is 0.0 at training time. Leave it on and the model is trained
against labels that are wrong on half its inputs.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from pipeline.detect import CANONICAL_FISH
from pipeline.landmarks import SCHEMA

ROBOFLOW_WORKSPACE = "fish-o3fkg"
ROBOFLOW_PROJECT = "fishkeypoints"


# ---------------------------------------------------------------------------
# the real path
# ---------------------------------------------------------------------------


def fetch_roboflow(
    out_dir: str | Path,
    api_key: str | None = None,
    workspace: str = ROBOFLOW_WORKSPACE,
    project: str = ROBOFLOW_PROJECT,
    version: int | None = None,
    fmt: str = "yolov8",
) -> Path:
    """Download the fishKeypoints export. Needs ROBOFLOW_API_KEY in the env.

    Deliberately uses urllib and the documented REST endpoints rather than the
    `roboflow` pip package: it's two requests, and it avoids a dependency whose
    only job would be to make those two requests.
    """
    import json
    import urllib.request
    import zipfile

    key = api_key or os.environ.get("ROBOFLOW_API_KEY") or os.environ.get("RF_KEY")
    if not key:
        raise RuntimeError(
            "no ROBOFLOW_API_KEY in the environment. Set it and re-run:\n"
            "  export ROBOFLOW_API_KEY=...\n"
            "  python -m eval.dataset --fetch data/fishkeypoints\n"
            "Getting a key needs a Roboflow account, which I can't create."
        )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta_url = f"https://api.roboflow.com/{workspace}/{project}?api_key={key}"
    with urllib.request.urlopen(meta_url, timeout=60) as r:
        meta = json.loads(r.read())
    (out / "roboflow_project.json").write_text(json.dumps(meta, indent=2))

    versions = meta.get("versions") or []
    if version is None:
        if not versions:
            raise RuntimeError("the project metadata lists no versions to export")
        version = versions[-1].get("id", "").split("/")[-1] or 1

    export_url = (
        f"https://api.roboflow.com/{workspace}/{project}/{version}/{fmt}?api_key={key}"
    )
    with urllib.request.urlopen(export_url, timeout=120) as r:
        export = json.loads(r.read())
    link = export.get("export", {}).get("link")
    if not link:
        raise RuntimeError(f"no download link in the export response: {export}")

    zip_path = out / "export.zip"
    urllib.request.urlretrieve(link, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out)
    zip_path.unlink()

    print(f"downloaded {workspace}/{project} v{version} to {out}")
    print("NEXT: read the keypoint schema before writing any measurement code —")
    print("      docs/DATASETS.md has the gap analysis this has to be checked against.")
    return out


def describe_schema(dataset_dir: str | Path) -> dict[str, Any]:
    """Read whatever a YOLO-pose export says about its keypoints.

    The whole point of checkpoint 1 is not assuming. This prints what the export
    actually declares — keypoint count, names if it has them, class names — so
    the schema in pipeline/landmarks.py can be checked against it rather than
    hoped at.
    """
    d = Path(dataset_dir)
    candidates = list(d.glob("**/data.yaml")) + list(d.glob("**/*.yaml"))
    if not candidates:
        raise FileNotFoundError(f"no dataset yaml under {d}")
    doc = yaml.safe_load(candidates[0].read_text())
    return {
        "yaml": str(candidates[0]),
        "kpt_shape": doc.get("kpt_shape"),
        "keypoint_names": doc.get("kpt_names") or doc.get("keypoints"),
        "flip_idx": doc.get("flip_idx"),
        "classes": doc.get("names"),
        "raw": doc,
    }


# ---------------------------------------------------------------------------
# the synthetic stand-in
# ---------------------------------------------------------------------------


def _render_fish(
    size: tuple[int, int], rng: random.Random
) -> tuple[np.ndarray, np.ndarray]:
    """One image with one geometric fish. Returns (image, keypoints in pixels)."""
    h, w = size
    bg = rng.randint(20, 80)
    img = np.full((h, w, 3), bg, np.uint8)
    noise = np.random.default_rng(rng.randrange(1 << 30)).normal(0, 8, (h, w, 3))
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    length = rng.uniform(0.35, 0.75) * w
    angle = rng.uniform(-25, 25)
    bend = rng.uniform(-0.05, 0.05)
    cx = rng.uniform(0.35, 0.65) * w
    cy = rng.uniform(0.35, 0.65) * h

    t = np.deg2rad(angle)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    pts: dict[str, np.ndarray] = {}
    for name, (fx, fy) in CANONICAL_FISH.items():
        fy = fy + bend * np.sin(np.pi * fx)
        pts[name] = np.array([cx, cy]) + R @ (np.array([fx - 0.5, fy]) * length)

    body_colour = tuple(int(c) for c in np.clip(
        np.array([160, 170, 175]) + rng.uniform(-40, 40), 40, 245))
    outline = [
        "snout_tip", "dorsal_origin", "dorsal_apex", "dorsal_insertion",
        "peduncle_dorsal", "caudal_tip", "peduncle_ventral", "ventral_margin",
    ]
    poly = np.array([pts[n] for n in outline], dtype=np.int32)
    cv2.fillPoly(img, [poly], body_colour)
    cv2.circle(img, pts["eye_anterior"].astype(int), max(2, int(length * 0.012)), (20, 20, 20), -1)

    kp = np.array([pts[n] for n in SCHEMA], dtype=np.float64)
    return img, kp


def make_synthetic(
    out_dir: str | Path, n_train: int = 120, n_val: int = 30, size: int = 640, seed: int = 0
) -> Path:
    """Write a YOLO-pose dataset of geometric fish. See the module docstring for
    what this is and is not for."""
    out = Path(out_dir)
    rng = random.Random(seed)

    for split, n in (("train", n_train), ("val", n_val)):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img, kp = _render_fish((size, size), rng)
            stem = f"{split}_{i:05d}"
            cv2.imwrite(str(out / "images" / split / f"{stem}.jpg"), img)

            x1, y1 = kp.min(axis=0)
            x2, y2 = kp.max(axis=0)
            pad = 0.02 * size
            x1, y1 = max(x1 - pad, 0), max(y1 - pad, 0)
            x2, y2 = min(x2 + pad, size - 1), min(y2 + pad, size - 1)
            cx, cy = (x1 + x2) / 2 / size, (y1 + y2) / 2 / size
            bw, bh = (x2 - x1) / size, (y2 - y1) / size

            fields = [f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"]
            for x, y in kp:
                fields.append(f"{x / size:.6f} {y / size:.6f} 2")  # 2 = labelled, visible
            (out / "labels" / split / f"{stem}.txt").write_text(" ".join(fields) + "\n")

    data_yaml = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "fish"},
        "kpt_shape": [len(SCHEMA), 3],
        # Identity. A horizontal flip of a fish in lateral view maps snout to
        # tail and there is no permutation that fixes that — see the module
        # docstring. eval/train.py also sets fliplr=0.0.
        "flip_idx": list(range(len(SCHEMA))),
        "kpt_names": list(SCHEMA),
    }
    yaml_path = out / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    print(f"wrote {n_train} train / {n_val} val synthetic images to {out}")
    print("these are polygons, not fish. Metrics from them mean nothing about fish.")
    return yaml_path


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m eval.dataset")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fetch", metavar="OUT_DIR", help="pull fishKeypoints from Roboflow")
    g.add_argument("--synthetic", metavar="OUT_DIR", help="write the synthetic stand-in")
    g.add_argument("--describe", metavar="DATASET_DIR", help="print a dataset's keypoint schema")
    ap.add_argument("--train-n", type=int, default=120)
    ap.add_argument("--val-n", type=int, default=30)
    args = ap.parse_args(argv)

    if args.fetch:
        try:
            fetch_roboflow(args.fetch)
        except RuntimeError as exc:
            print(str(exc))
            return 1
    elif args.synthetic:
        make_synthetic(args.synthetic, args.train_n, args.val_n)
    else:
        import json

        print(json.dumps(describe_schema(args.describe), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
