"""Load config.yaml. That's the whole module.

It lives in pipeline/ rather than at the repo root because pipeline/ has to be
importable as a library on its own, see the note in the spec about not
importing api/ from here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read config.yaml into a plain dict. No schema validation, no defaults
    injection. If a key is missing the caller finds out at the point of use,
    which is where the error message is most useful."""
    p = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(p, "r") as f:
        return yaml.safe_load(f)


def resolve_path(value: str | Path) -> Path:
    """Turn a config path into an absolute one, relative to the repo root."""
    p = Path(value)
    return p if p.is_absolute() else REPO_ROOT / p
