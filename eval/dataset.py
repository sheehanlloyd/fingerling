"""Dataset plumbing for Phase 2: the Roboflow pull, and a synthetic stand-in.

Two things live here.

`fetch_roboflow` is the real path. It is written and it does not work tonight,
because it needs a ROBOFLOW_API_KEY and there isn't one in the environment.
Creating an account to get one is off the table, so this function exists ready to
run the moment a key shows up. Nothing about it is speculative. The endpoints
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

# Fish Measurement, by Fish Count. One salmonid parr per frame in a tray, four
# keypoints, CC BY 4.0. NOT fishKeypoints, which was the checkpoint 1 pick and
# turned out to be aerial drone footage of wild schools, see docs/DATASETS.md.
ROBOFLOW_WORKSPACE = "fish-count"
ROBOFLOW_PROJECT = "fish-measurement-z2ois"


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
    export_timeout_s: float = 600.0,
) -> Path:
    """Download a Roboflow YOLO-pose export. Needs ROBOFLOW_API_KEY in the env.

    Deliberately uses urllib and the documented REST endpoints rather than the
    `roboflow` pip package: it's two requests, and it avoids a dependency whose
    only job would be to make those two requests.
    """
    import json
    import time
    import urllib.request
    import zipfile

    key = api_key or os.environ.get("ROBOFLOW_API_KEY") or os.environ.get("RF_KEY")
    if not key:
        raise RuntimeError(
            "no ROBOFLOW_API_KEY in the environment. Set it and re-run:\n"
            "  export ROBOFLOW_API_KEY=...\n"
            "  python -m eval.dataset --fetch data/fish-measurement\n"
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
        # Roboflow returns versions newest-first. The original code took
        # versions[-1], which is the OLDEST (v1, 55 images) and no generated
        # export at all, which is why this failed with {'progress': 0} the first
        # time it was ever run for real. Pick the newest version that has
        # finished generating, and say which one out loud.
        ready = [
            v for v in versions
            if not v.get("generating", False) and v.get("progress", 0) >= 1
        ]
        if not ready:
            raise RuntimeError("the project metadata lists no finished versions to export")
        newest = max(ready, key=lambda v: v.get("created", 0))
        version = str(newest.get("id", "")).split("/")[-1] or "1"
        print(
            f"selected version {version} "
            f"({newest.get('images')} images, generated {newest.get('name')})"
        )

    # An export has to be built server-side before there's a link. The endpoint
    # returns {"progress": <float>} while it's still working, so poll rather than
    # treating the first miss as a failure.
    export_url = (
        f"https://api.roboflow.com/{workspace}/{project}/{version}/{fmt}?api_key={key}"
    )
    link = None
    deadline = time.monotonic() + export_timeout_s
    while time.monotonic() < deadline:
        with urllib.request.urlopen(export_url, timeout=120) as r:
            export = json.loads(r.read())
        link = export.get("export", {}).get("link")
        if link:
            break
        pct = float(export.get("progress", 0.0)) * 100
        print(f"  roboflow is generating the {fmt} export... {pct:.0f}%")
        time.sleep(5.0)
    if not link:
        raise RuntimeError(
            f"export for {workspace}/{project} v{version} did not become available "
            f"within {export_timeout_s:.0f}s. Last response: {export}"
        )

    zip_path = out / "export.zip"
    urllib.request.urlretrieve(link, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out)
    zip_path.unlink()

    print(f"downloaded {workspace}/{project} v{version} to {out}")
    print("NEXT: read the keypoint schema before writing any measurement code:")
    print("      docs/DATASETS.md has the gap analysis this has to be checked against.")
    return out


def describe_schema(dataset_dir: str | Path) -> dict[str, Any]:
    """Read whatever a YOLO-pose export says about its keypoints.

    The whole point of checkpoint 1 is not assuming. This prints what the export
    actually declares: keypoint count, names if it has them, class names, so
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
# re-splitting, because the published split leaks
# ---------------------------------------------------------------------------


def source_stem(path: str | Path) -> str:
    """The original photograph a Roboflow file came from.

    Roboflow names exports `<source>_jpg.rf.<hash>.jpg`, one file per augmented
    copy. Everything before `_jpg.rf.` identifies the photograph the copy was
    made from.
    """
    return Path(path).name.split("_jpg.rf.")[0]


def audit_split_leakage(dataset_dir: str | Path) -> dict[str, Any]:
    """Count source photographs that appear in more than one split.

    Worth running on any Roboflow export before believing a validation number.
    On Fish Measurement v9 this returns 34 leaked sources out of 120: the
    published split was made over the 245 augmented FILES, not over the 120
    photographs behind them, so rotated and re-exposed copies of the same fish
    sit in train and test at once. The files aren't byte-identical, so a hash
    check finds nothing and everything looks fine.

    The consequence is not subtle. Trained on the published split this model
    scored pose mAP50-95 0.953 on "held-out" test data that was nothing of the
    kind.
    """
    d = Path(dataset_dir)
    splits: dict[str, set[str]] = {}
    for sp in ("train", "valid", "test"):
        for img in (d / sp / "images").glob("*.jpg"):
            splits.setdefault(source_stem(img), set()).add(sp)
    leaked = {k: sorted(v) for k, v in splits.items() if len(v) > 1}
    return {
        "unique_sources": len(splits),
        "leaked_sources": len(leaked),
        "leaked": leaked,
    }


def regroup_split(
    dataset_dir: str | Path,
    out_dir: str | Path,
    fractions: tuple[float, float, float] = (0.70, 0.18, 0.12),
    seed: int = 0,
) -> Path:
    """Re-split a Roboflow export by SOURCE PHOTOGRAPH instead of by file.

    Every augmented copy of one photograph lands in the same split, so a
    validation score means what it's supposed to mean. This is a group split,
    the standard fix; the only fiddly part is that the grouping key has to be
    recovered from the filename because the export doesn't record it.

    The split sizes come out approximate, because sources carry between one and
    four copies each and the groups are assigned whole.
    """
    import shutil

    src = Path(dataset_dir)
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)

    by_source: dict[str, list[tuple[Path, Path]]] = {}
    for sp in ("train", "valid", "test"):
        for img in sorted((src / sp / "images").glob("*.jpg")):
            lbl = src / sp / "labels" / (img.stem + ".txt")
            if lbl.exists():
                by_source.setdefault(source_stem(img), []).append((img, lbl))

    sources = sorted(by_source)
    random.Random(seed).shuffle(sources)
    n = len(sources)
    n_train = int(round(fractions[0] * n))
    n_valid = int(round(fractions[1] * n))
    groups = {
        "train": sources[:n_train],
        "valid": sources[n_train:n_train + n_valid],
        "test": sources[n_train + n_valid:],
    }

    for sp, names in groups.items():
        (out / sp / "images").mkdir(parents=True, exist_ok=True)
        (out / sp / "labels").mkdir(parents=True, exist_ok=True)
        for name in names:
            for img, lbl in by_source[name]:
                shutil.copy2(img, out / sp / "images" / img.name)
                shutil.copy2(lbl, out / sp / "labels" / lbl.name)

    doc = yaml.safe_load((src / "data.yaml").read_text())
    doc.update({"train": "../train/images", "val": "../valid/images", "test": "../test/images"})
    doc["regrouped"] = {
        "note": "re-split by source photograph; the published split leaked across splits",
        "seed": seed,
        "sources": {sp: len(v) for sp, v in groups.items()},
    }
    (out / "data.yaml").write_text(yaml.safe_dump(doc, sort_keys=False))

    for sp, names in groups.items():
        files = sum(len(by_source[x]) for x in names)
        print(f"  {sp:<6} {len(names):>4} sources  {files:>4} files")
    return out


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
        # tail and there is no permutation that fixes that, see the module
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
    g.add_argument("--fetch", metavar="OUT_DIR", help="pull the dataset from Roboflow")
    g.add_argument("--synthetic", metavar="OUT_DIR", help="write the synthetic stand-in")
    g.add_argument("--describe", metavar="DATASET_DIR", help="print a dataset's keypoint schema")
    g.add_argument("--regroup", metavar="DATASET_DIR",
                   help="re-split a downloaded dataset by SOURCE PHOTOGRAPH. The published "
                        "Roboflow split is over augmented files, so copies of one photo land "
                        "in train and test at once, see docs/DATASETS.md")
    ap.add_argument("--workspace", default=ROBOFLOW_WORKSPACE,
                    help=f"Roboflow workspace (default: {ROBOFLOW_WORKSPACE})")
    ap.add_argument("--project", default=ROBOFLOW_PROJECT,
                    help=f"Roboflow project (default: {ROBOFLOW_PROJECT})")
    ap.add_argument("--version", type=int, default=None,
                    help="dataset version; default is the newest that has finished generating")
    ap.add_argument("--out", metavar="OUT_DIR", help="destination for --regroup")
    ap.add_argument("--train-n", type=int, default=120)
    ap.add_argument("--val-n", type=int, default=30)
    args = ap.parse_args(argv)

    if args.fetch:
        try:
            fetch_roboflow(
                args.fetch,
                workspace=args.workspace,
                project=args.project,
                version=args.version,
            )
        except RuntimeError as exc:
            print(str(exc))
            return 1
    elif args.synthetic:
        make_synthetic(args.synthetic, args.train_n, args.val_n)
    elif args.regroup:
        audit = audit_split_leakage(args.regroup)
        print(
            f"{audit['unique_sources']} source photographs, "
            f"{audit['leaked_sources']} of them appear in more than one split"
        )
        out = args.out or (str(args.regroup).rstrip("/") + "-grouped")
        regroup_split(args.regroup, out)
        after = audit_split_leakage(out)
        print(f"after regrouping: {after['leaked_sources']} leaked (must be 0) -> {out}")
    else:
        import json

        print(json.dumps(describe_schema(args.describe), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
