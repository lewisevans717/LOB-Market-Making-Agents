"""Helpers for loading BSE module code from a pinned git submodule."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType


_CACHE: ModuleType | None = None


def bse_root_path() -> Path:
    """Return the path to the BSE submodule checkout."""
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "third_party" / "BristolStockExchange"


def load_bse_module() -> ModuleType:
    """Load `BSE.py` from the vendored submodule."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    bse_path = bse_root_path() / "BSE.py"
    if not bse_path.exists():
        raise RuntimeError(
            "BSE source not found. Initialize submodules with: "
            "`git submodule update --init --recursive`."
        )

    spec = importlib.util.spec_from_file_location("lobmm_bse_module", bse_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load BSE module spec from: {bse_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _CACHE = module
    return module


def get_bse_commit() -> str:
    """Return the pinned BSE submodule commit hash if available."""
    root = bse_root_path()
    if not root.exists():
        return "missing"
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"
    return commit or "unknown"
