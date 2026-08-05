"""Explicit Office-Home acquisition helpers for official and allowed fallback sources."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import zipfile

from .common import (
    HF_DATASET,
    OFFICIAL_DRIVE_URL,
    atomic_write_json,
    sha256_file,
    utc_now,
)
from .manifest import locate_dataset_root


ARCHIVE_NAME = "OfficeHomeDataset_10072016.zip"


def _safe_extract_zip(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            member_path = (destination / member.filename).resolve()
            if member_path != destination and destination not in member_path.parents:
                raise ValueError(f"Unsafe path in Office-Home archive: {member.filename!r}")
        bundle.extractall(destination)


def _existing_dataset(data_root: Path) -> Path | None:
    try:
        root, _ = locate_dataset_root(data_root)
    except (FileNotFoundError, ValueError):
        return None
    return root


def download_official(data_root: str | Path) -> dict:
    """Download the official Google Drive archive, or reuse a valid extraction."""
    data_root = Path(data_root).expanduser().resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    existing = _existing_dataset(data_root)
    if existing is not None:
        marker_path = data_root / ".dataset_source.json"
        if marker_path.exists():
            with marker_path.open("r", encoding="utf-8") as handle:
                marker = json.load(handle)
            if marker.get("provider") != "official":
                raise RuntimeError(
                    f"Existing dataset at {existing} is marked as provider "
                    f"{marker.get('provider')!r}, not official."
                )
        return {
            "provider": "official",
            "source_url": OFFICIAL_DRIVE_URL,
            "dataset_root": str(existing),
            "download_skipped": True,
        }

    archive = data_root / ARCHIVE_NAME
    if not archive.exists():
        try:
            import gdown
        except ImportError as exc:
            raise RuntimeError(
                "Official download requires gdown. Install requirements.txt, then retry."
            ) from exc
        # Avoid gdown's global ~/.cache cookie jar: experiment environments can
        # legitimately have a read-only home cache, and this public file does
        # not require an authenticated cookie.
        result = gdown.download(
            url=OFFICIAL_DRIVE_URL,
            output=str(archive),
            quiet=False,
            fuzzy=True,
            use_cookies=False,
        )
        if result is None or not archive.is_file() or archive.stat().st_size == 0:
            raise RuntimeError(
                "Official Google Drive download did not produce the archive. "
                "Retry manually or explicitly use --source huggingface."
            )
    if not zipfile.is_zipfile(archive):
        raise RuntimeError(f"Official download is not a valid ZIP archive: {archive}")

    _safe_extract_zip(archive, data_root)
    dataset_root, _ = locate_dataset_root(data_root)
    marker = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "provider": "official",
        "source_url": OFFICIAL_DRIVE_URL,
        "archive_path": str(archive),
        "archive_size_bytes": int(archive.stat().st_size),
        "archive_sha256": sha256_file(archive),
        "dataset_root": str(dataset_root),
        "download_skipped": False,
    }
    atomic_write_json(data_root / ".dataset_source.json", marker)
    return marker


def _extension_from_image_record(image: dict, fallback: str = ".jpg") -> str:
    path = image.get("path")
    if path:
        suffix = Path(path).suffix.casefold()
        if suffix:
            return suffix
    return fallback


def download_huggingface(data_root: str | Path) -> dict:
    """Materialize the explicitly allowed Hugging Face fallback without re-encoding bytes."""
    data_root = Path(data_root).expanduser().resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    existing = _existing_dataset(data_root)
    if existing is not None:
        marker_path = data_root / ".dataset_source.json"
        if marker_path.exists():
            with marker_path.open("r", encoding="utf-8") as handle:
                marker = json.load(handle)
            if marker.get("provider") != "huggingface":
                raise RuntimeError(
                    f"Existing dataset at {existing} is not marked as the Hugging Face fallback."
                )
        return {
            "provider": "huggingface",
            "source_url": f"https://huggingface.co/datasets/{HF_DATASET}",
            "dataset_root": str(existing),
            "download_skipped": True,
        }

    hf_home = data_root / ".hf-cache"
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))
    try:
        from datasets import Image as HFImage, load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face fallback requires requirements-hf-fallback.txt."
        ) from exc

    print(
        "WARNING: the official Office-Home archive is unavailable; explicitly using "
        f"the allowed Hugging Face fallback {HF_DATASET}.",
        flush=True,
    )
    dataset = load_dataset(HF_DATASET, split="train")
    label_feature = dataset.features["label"]
    label_names = list(getattr(label_feature, "names", []))
    if len(label_names) != 65:
        raise ValueError(f"Hugging Face fallback exposes {len(label_names)} labels, expected 65.")
    dataset = dataset.cast_column("image", HFImage(decode=False))
    output_root = data_root / "office_home_huggingface"
    if output_root.exists():
        raise FileExistsError(
            f"Partial Hugging Face output exists at {output_root}; inspect it before retrying."
        )
    output_root.mkdir(parents=True)

    try:
        for index, record in enumerate(dataset):
            domain = str(record["domain"])
            class_name = label_names[int(record["label"])]
            image = record["image"]
            suffix = _extension_from_image_record(image)
            destination = output_root / domain / class_name / f"row_{index:06d}{suffix}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            data = image.get("bytes")
            if data is not None:
                destination.write_bytes(data)
            elif image.get("path"):
                shutil.copyfile(image["path"], destination)
            else:
                raise ValueError(f"Fallback row {index} has neither image bytes nor a path.")
    except Exception:
        # Preserve the partial directory for audit instead of deleting downloaded data.
        raise

    dataset_root, _ = locate_dataset_root(data_root)
    marker = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "provider": "huggingface",
        "source_url": f"https://huggingface.co/datasets/{HF_DATASET}",
        "huggingface_dataset": HF_DATASET,
        "huggingface_fingerprint": getattr(dataset, "_fingerprint", None),
        "num_rows_reported": int(len(dataset)),
        "dataset_root": str(dataset_root),
        "download_skipped": False,
        "fallback_notice": (
            "The official Google Drive archive was unavailable. This run uses the explicitly "
            "allowed flwrlabs/office-home fallback and validates its contents independently."
        ),
    }
    atomic_write_json(data_root / ".dataset_source.json", marker)
    return marker


def acquire_dataset(data_root: str | Path, source: str = "official") -> dict:
    if source == "official":
        return download_official(data_root)
    if source == "huggingface":
        return download_huggingface(data_root)
    raise ValueError("source must be 'official' or 'huggingface'.")
