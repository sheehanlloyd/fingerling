"""Phase 2 training: fine-tune a YOLO-pose model on this laptop.

    python -m eval.train --data data/fish-measurement-grouped/data.yaml --epochs 100

Everything about this is shaped by the machine (M4 Pro, 24 GB, MPS) and by the
dataset being small (fishKeypoints is 593 images).

  amp=False. Mixed precision is unreliable on MPS with Ultralytics, and that
  matches what I'd expect, since the MPS autocast path is the
  least-travelled one in that codebase. Not negotiable, and it's a flag here
  rather than a default so it can't be lost.

  yolo11n-pose, the smallest variant. On ~600 images anything bigger is memorising
  rather than learning; the nano model has about 3M parameters, which is already
  generous for one class and twelve keypoints. Going up a size is a one-word
  change if the data ever justifies it.

  fliplr=0.0. The important one. Ultralytics flips images horizontally by default
  and permutes keypoints to match. For a fish in lateral view a flip maps the
  snout onto the tail and no permutation of these landmarks expresses that, so
  the augmentation would train against wrong labels on half the inputs. See
  eval/dataset.py.

  Everything the run produces (the actual epoch count, the actual wall time, the
  metrics Ultralytics reports) gets written to a JSON file next to the weights.
  No number in this repo's documentation should come from anywhere else.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "yolo11n-pose.pt"


def device_report() -> dict[str, Any]:
    import torch

    return {
        "torch": torch.__version__,
        "mps_available": bool(torch.backends.mps.is_available()),
        "mps_built": bool(torch.backends.mps.is_built()),
    }


def train(
    data: str | Path,
    model: str = DEFAULT_MODEL,
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 8,
    device: str = "mps",
    project: str = "runs/pose",
    name: str = "fingerling",
    patience: int = 25,
    seed: int = 0,
) -> dict[str, Any]:
    from ultralytics import YOLO

    dev = device_report()
    if device == "mps" and not dev["mps_available"]:
        print("MPS is not available; falling back to cpu. This will be slow.")
        device = "cpu"

    # Resolve `project` to an absolute path before handing it over. Ultralytics
    # treats a RELATIVE project as relative to its own global runs_dir (a
    # user-level settings.json, not anything in this repo) with the task name
    # inserted, so `project="runs/pose"` came back as
    # `runs/pose/runs/pose/smoke`. An absolute path is used verbatim, which also
    # means the directory you asked for is the one you get regardless of what
    # that global settings file happens to say on someone else's machine.
    project = str(Path(project).resolve())

    yolo = YOLO(model)
    t0 = time.perf_counter()
    results = yolo.train(
        data=str(data),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
        patience=patience,
        seed=seed,
        # See the module docstring. Both of these are the point of this file.
        amp=False,
        fliplr=0.0,
        verbose=True,
    )
    wall = time.perf_counter() - t0

    save_dir = Path(getattr(results, "save_dir", Path(project) / name))
    metrics = _extract_metrics(results)
    summary = {
        "data": str(data),
        "base_model": model,
        "epochs_requested": epochs,
        "epochs_completed": _epochs_completed(save_dir),
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "amp": False,
        "fliplr": 0.0,
        "wall_seconds": round(wall, 1),
        "weights": str(save_dir / "weights" / "best.pt"),
        "device_report": dev,
        "metrics": metrics,
    }
    (save_dir / "fingerling_train_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def _extract_metrics(results: Any) -> dict[str, Any]:
    """Whatever Ultralytics actually reported, verbatim. No rounding up, no
    picking the flattering number. The point of this file is that the README
    can only quote things that came out of here."""
    out: dict[str, Any] = {}
    box = getattr(results, "results_dict", None)
    if isinstance(box, dict):
        out.update({k: (float(v) if isinstance(v, (int, float)) else str(v)) for k, v in box.items()})
    return out


def _epochs_completed(save_dir: Path) -> int | None:
    """Count rows in results.csv. Early stopping means "epochs" in the config is
    a request, not a fact, and the fact is what belongs in the docs."""
    csv = save_dir / "results.csv"
    if not csv.exists():
        return None
    lines = [ln for ln in csv.read_text().splitlines() if ln.strip()]
    return max(len(lines) - 1, 0)


def validate(weights: str | Path, data: str | Path, device: str = "mps") -> dict[str, Any]:
    """Run the validation split and report the pose metrics as-reported."""
    from ultralytics import YOLO

    m = YOLO(str(weights))
    r = m.val(data=str(data), device=device, verbose=False)
    return {k: (float(v) if isinstance(v, (int, float)) else str(v))
            for k, v in getattr(r, "results_dict", {}).items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.train")
    ap.add_argument("--data", required=True, help="path to the dataset data.yaml")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--project", default="runs/pose")
    ap.add_argument("--name", default="fingerling")
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--validate-only", default=None, metavar="WEIGHTS")
    args = ap.parse_args(argv)

    if args.validate_only:
        print(json.dumps(validate(args.validate_only, args.data, args.device), indent=2))
        return 0

    train(
        data=args.data, model=args.model, epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, device=args.device, project=args.project,
        name=args.name, patience=args.patience,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
