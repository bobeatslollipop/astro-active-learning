"""Shared constants, deterministic runtime helpers, and artifact I/O."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import tempfile
from typing import Any

import numpy as np
import pandas as pd


CANONICAL_DOMAINS = ("art", "clipart", "product", "real_world")
NUM_CLASSES = 65
DEFAULT_SEED = 0
EXPECTED_DOMAIN_COUNTS = {
    "art": 2427,
    "clipart": 4365,
    "product": 4439,
    "real_world": 4357,
}
OFFICIAL_PAGE = "https://www.hemanthdv.org/officeHomeDataset.html"
OFFICIAL_DRIVE_URL = (
    "https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view"
    "?resourcekey=0-2SNWq0CDAuWOBRRBL7ZZsw&usp=sharing"
)
HF_DATASET = "flwrlabs/office-home"
IMAGE_EXTENSIONS = {
    ".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_domain_name(value: str) -> str:
    """Return the canonical Office-Home domain name or raise."""
    token = "".join(ch for ch in value.casefold() if ch.isalnum())
    aliases = {
        "art": "art",
        "artistic": "art",
        "clipart": "clipart",
        "product": "product",
        "products": "product",
        "realworld": "real_world",
        "real": "real_world",
    }
    if token not in aliases:
        raise ValueError(f"Unknown Office-Home domain name: {value!r}")
    return aliases[token]


def resolve_data_root(value: str | os.PathLike[str] | None = None) -> Path:
    if value is not None:
        return Path(value).expanduser().resolve()
    env_value = os.environ.get("OFFICEHOME_ROOT")
    if env_value:
        return Path(env_value).expanduser().resolve()
    return (Path(__file__).resolve().parents[1] / ".data").resolve()


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def atomic_write_json(path: str | os.PathLike[str], payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    tmp_path = Path(handle.name)
    try:
        with handle:
            json.dump(_json_ready(payload), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def atomic_write_csv(path: str | os.PathLike[str], frame: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    tmp_path = Path(handle.name)
    try:
        with handle:
            frame.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def set_deterministic(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (AttributeError, TypeError):
        pass


def runtime_metadata() -> dict[str, Any]:
    result: dict[str, Any] = {
        "created_at_utc": utc_now(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch

        result.update({
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "cudnn": None if not torch.backends.cudnn.is_available() else str(torch.backends.cudnn.version()),
        })
        if torch.cuda.is_available():
            result["cuda_device_count"] = int(torch.cuda.device_count())
            result["cuda_device_names"] = [
                torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
            ]
    except ImportError:
        result["torch"] = None
    try:
        import torchvision

        result["torchvision"] = torchvision.__version__
    except ImportError:
        result["torchvision"] = None
    for package, import_name in (
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("Pillow", "PIL"),
        ("scikit-learn", "sklearn"),
    ):
        try:
            module = __import__(import_name)
            result[package] = getattr(module, "__version__", "unknown")
        except ImportError:
            result[package] = None
    return result


def resolve_device(value: str):
    import torch

    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but torch.cuda.is_available() is false: {value}")
    return device


def load_single_column_ids(path: str | os.PathLike[str], column: str = "row_id") -> np.ndarray:
    frame = pd.read_csv(path)
    if list(frame.columns) != [column]:
        raise ValueError(f"{path} must contain exactly one column named {column!r}.")
    if frame[column].isna().any():
        raise ValueError(f"{path} contains missing {column} values.")
    ids = frame[column].to_numpy(dtype=np.int64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"{path} contains duplicate {column} values.")
    return ids
