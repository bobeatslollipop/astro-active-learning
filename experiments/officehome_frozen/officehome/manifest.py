"""Office-Home discovery, validation, deterministic manifests, and task splits."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from sklearn.model_selection import StratifiedShuffleSplit

from .common import (
    CANONICAL_DOMAINS,
    EXPECTED_DOMAIN_COUNTS,
    HF_DATASET,
    IMAGE_EXTENSIONS,
    NUM_CLASSES,
    OFFICIAL_DRIVE_URL,
    OFFICIAL_PAGE,
    atomic_write_csv,
    atomic_write_json,
    normalize_domain_name,
    sha256_file,
    utc_now,
)


PRIVATE_COLUMNS = ["row_id", "relative_image_path", "domain", "class_name", "class_id"]
FEATURE_MANIFEST_COLUMNS = ["row_id", "relative_image_path", "domain"]


def _domain_directories(candidate: Path) -> dict[str, Path] | None:
    found: dict[str, Path] = {}
    if not candidate.is_dir():
        return None
    for child in candidate.iterdir():
        if not child.is_dir():
            continue
        try:
            canonical = normalize_domain_name(child.name)
        except ValueError:
            continue
        if canonical in found:
            raise ValueError(
                f"Multiple directories map to domain {canonical!r}: {found[canonical]} and {child}"
            )
        found[canonical] = child
    return found if set(found) == set(CANONICAL_DOMAINS) else None


def locate_dataset_root(data_root: str | Path) -> tuple[Path, dict[str, Path]]:
    """Locate the directory whose immediate children are the four domains."""
    data_root = Path(data_root).expanduser().resolve()
    candidates = [data_root]
    if data_root.is_dir():
        candidates.extend(sorted((p for p in data_root.iterdir() if p.is_dir()), key=str))
        for child in list(candidates[1:]):
            candidates.extend(sorted((p for p in child.iterdir() if p.is_dir()), key=str))
    matches: list[tuple[Path, dict[str, Path]]] = []
    for candidate in candidates:
        mapped = _domain_directories(candidate)
        if mapped is not None:
            matches.append((candidate, mapped))
    unique = {str(path): (path, mapped) for path, mapped in matches}
    if not unique:
        raise FileNotFoundError(
            f"Could not find the four Office-Home domain directories below {data_root}."
        )
    if len(unique) > 1:
        roots = ", ".join(sorted(unique))
        raise ValueError(f"Multiple Office-Home dataset roots found below {data_root}: {roots}")
    return next(iter(unique.values()))


def _validate_image(path: Path) -> None:
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    with Image.open(path) as image:
        image.load()
        image.convert("RGB").load()


def _class_directories(domain_dir: Path) -> dict[str, Path]:
    result = {child.name: child for child in domain_dir.iterdir() if child.is_dir()}
    if not result:
        raise ValueError(f"Domain directory has no class subdirectories: {domain_dir}")
    return result


def _source_metadata(data_root: Path, requested_source: str) -> dict:
    marker = data_root / ".dataset_source.json"
    if marker.exists():
        with marker.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("provider") != requested_source:
            raise ValueError(
                f"Requested dataset source {requested_source!r} does not match source marker "
                f"provider {payload.get('provider')!r}."
            )
        payload["marker_path"] = str(marker)
        return payload
    return {
        "provider": requested_source,
        "source_url": OFFICIAL_DRIVE_URL if requested_source == "official" else HF_DATASET,
        "source_marker_missing": True,
    }


def build_manifest(
    data_root: str | Path,
    output_dir: str | Path,
    *,
    dataset_source: str = "official",
    validate_images: bool = True,
) -> dict:
    """Build and atomically save the deterministic global manifest."""
    if dataset_source not in {"official", "huggingface"}:
        raise ValueError("dataset_source must be 'official' or 'huggingface'.")
    data_root = Path(data_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    dataset_root, domain_dirs = locate_dataset_root(data_root)

    class_dirs = {domain: _class_directories(domain_dirs[domain]) for domain in CANONICAL_DOMAINS}
    class_sets = {domain: set(mapping) for domain, mapping in class_dirs.items()}
    reference = class_sets[CANONICAL_DOMAINS[0]]
    mismatches = {
        domain: {
            "missing": sorted(reference - names),
            "extra": sorted(names - reference),
        }
        for domain, names in class_sets.items()
        if names != reference
    }
    if mismatches:
        raise ValueError(f"Office-Home class sets differ across domains: {mismatches}")
    if len(reference) != NUM_CLASSES:
        raise ValueError(f"Expected {NUM_CLASSES} shared classes, found {len(reference)}.")

    class_names = sorted(reference, key=lambda value: (value.casefold(), value))
    name_to_id = {name: index for index, name in enumerate(class_names)}
    records: list[dict] = []
    resolved_paths: set[Path] = set()
    unreadable: list[str] = []

    for domain in CANONICAL_DOMAINS:
        for class_name in class_names:
            class_dir = class_dirs[domain][class_name]
            images = sorted(
                (
                    path for path in class_dir.rglob("*")
                    if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
                ),
                key=lambda path: path.relative_to(dataset_root).as_posix().casefold(),
            )
            if not images:
                raise ValueError(f"No supported images found in {class_dir}.")
            for image_path in images:
                resolved = image_path.resolve()
                if resolved in resolved_paths:
                    raise ValueError(f"Duplicate image path or symlink target: {image_path}")
                resolved_paths.add(resolved)
                if validate_images:
                    try:
                        _validate_image(image_path)
                    except Exception as exc:  # PIL exposes multiple decode exception types.
                        unreadable.append(f"{image_path}: {type(exc).__name__}: {exc}")
                        if len(unreadable) >= 20:
                            break
                records.append({
                    "relative_image_path": image_path.relative_to(dataset_root).as_posix(),
                    "domain": domain,
                    "class_name": class_name,
                    "class_id": name_to_id[class_name],
                })
            if unreadable:
                break
        if unreadable:
            break
    if unreadable:
        raise ValueError("Unreadable Office-Home images:\n" + "\n".join(unreadable))

    private = pd.DataFrame.from_records(records)
    if private["relative_image_path"].duplicated().any():
        duplicates = private.loc[
            private["relative_image_path"].duplicated(keep=False), "relative_image_path"
        ].tolist()
        raise ValueError(f"Duplicate relative image paths: {duplicates[:20]}")
    private.insert(0, "row_id", np.arange(len(private), dtype=np.int64))
    private = private[PRIVATE_COLUMNS]
    feature_manifest = private[FEATURE_MANIFEST_COLUMNS].copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    private_path = output_dir / "manifest_private.csv"
    feature_manifest_path = output_dir / "manifest.csv"
    mapping_path = output_dir / "class_mapping.json"
    metadata_path = output_dir / "dataset_metadata.json"
    atomic_write_csv(private_path, private)
    atomic_write_csv(feature_manifest_path, feature_manifest)
    atomic_write_json(mapping_path, {
        "schema_version": 1,
        "num_classes": NUM_CLASSES,
        "name_to_id": name_to_id,
        "id_to_name": class_names,
    })

    counts = {domain: int((private["domain"] == domain).sum()) for domain in CANONICAL_DOMAINS}
    differences = {
        domain: counts[domain] - EXPECTED_DOMAIN_COUNTS[domain] for domain in CANONICAL_DOMAINS
    }
    metadata = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "dataset": "Office-Home",
        "official_page": OFFICIAL_PAGE,
        "requested_dataset_source": dataset_source,
        "dataset_source": _source_metadata(data_root, dataset_source),
        "data_root": str(data_root),
        "dataset_root": str(dataset_root),
        "canonical_domains": list(CANONICAL_DOMAINS),
        "domain_directories": {domain: domain_dirs[domain].name for domain in CANONICAL_DOMAINS},
        "num_classes": NUM_CLASSES,
        "num_images": int(len(private)),
        "domain_counts": counts,
        "expected_approximate_domain_counts": EXPECTED_DOMAIN_COUNTS,
        "domain_count_differences": differences,
        "expected_total": int(sum(EXPECTED_DOMAIN_COUNTS.values())),
        "count_warning": None if not any(differences.values()) else (
            "Actual counts differ from the prompt's approximate reference counts; actual counts are authoritative."
        ),
        "validated_all_images": bool(validate_images),
        "artifacts": {},
    }
    for name, path in (
        ("manifest_private", private_path),
        ("feature_manifest", feature_manifest_path),
        ("class_mapping", mapping_path),
    ):
        metadata["artifacts"][name] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }
    atomic_write_json(metadata_path, metadata)
    return metadata


def _validate_private_manifest(frame: pd.DataFrame) -> None:
    missing = set(PRIVATE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Private manifest is missing columns: {sorted(missing)}")
    if frame["row_id"].duplicated().any():
        raise ValueError("Private manifest contains duplicate row_id values.")
    if frame["relative_image_path"].duplicated().any():
        raise ValueError("Private manifest contains duplicate relative_image_path values.")
    if set(frame["domain"].unique()) != set(CANONICAL_DOMAINS):
        raise ValueError("Private manifest does not contain exactly the four canonical domains.")
    if set(frame["class_id"].unique()) != set(range(NUM_CLASSES)):
        raise ValueError(f"Private manifest does not contain class IDs 0..{NUM_CLASSES - 1}.")


def make_task_split(
    manifest_private: str | Path,
    output_dir: str | Path,
    *,
    source_domain: str,
    target_domain: str,
    protocol: str = "heldout",
    seed: int = 0,
) -> dict:
    source_domain = normalize_domain_name(source_domain)
    target_domain = normalize_domain_name(target_domain)
    if source_domain == target_domain:
        raise ValueError("source_domain and target_domain must differ.")
    if protocol not in {"heldout", "transductive"}:
        raise ValueError("protocol must be 'heldout' or 'transductive'.")

    manifest_private = Path(manifest_private).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    frame = pd.read_csv(manifest_private)
    _validate_private_manifest(frame)
    source = frame.loc[frame["domain"] == source_domain, PRIVATE_COLUMNS].copy()
    target = frame.loc[frame["domain"] == target_domain, PRIVATE_COLUMNS].copy()
    source.sort_values("row_id", inplace=True, ignore_index=True)
    target.sort_values("row_id", inplace=True, ignore_index=True)

    if protocol == "heldout":
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        pool_idx, test_idx = next(splitter.split(np.zeros(len(target)), target["class_id"]))
        target_pool = target.iloc[np.sort(pool_idx)].copy()
        target_test = target.iloc[np.sort(test_idx)].copy()
        evaluation_scope = "target_test"
    else:
        target_pool = target.copy()
        target_test = target.copy()
        evaluation_scope = "target_full_and_unqueried"

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "source_labeled": output_dir / "source_labeled.csv",
        "target_pool_public": output_dir / "target_pool_public.csv",
        "target_pool_oracle_private": output_dir / "target_pool_oracle_private.csv",
        "target_test_private": output_dir / "target_test_private.csv",
    }
    atomic_write_csv(paths["source_labeled"], source)
    atomic_write_csv(
        paths["target_pool_public"],
        target_pool[["row_id", "relative_image_path", "domain"]],
    )
    atomic_write_csv(paths["target_pool_oracle_private"], target_pool)
    atomic_write_csv(paths["target_test_private"], target_test)

    def counts_by_class(part: pd.DataFrame) -> dict[str, int]:
        counts = Counter(int(value) for value in part["class_id"])
        return {str(index): counts[index] for index in range(NUM_CLASSES)}

    metadata = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "manifest_private": str(manifest_private),
        "manifest_private_sha256": sha256_file(manifest_private),
        "source_domain": source_domain,
        "target_domain": target_domain,
        "protocol": protocol,
        "seed": int(seed),
        "num_classes": NUM_CLASSES,
        "evaluation_scope": evaluation_scope,
        "counts": {
            "source": int(len(source)),
            "target_pool": int(len(target_pool)),
            "target_test": int(len(target_test)),
        },
        "class_counts": {
            "source": counts_by_class(source),
            "target_pool": counts_by_class(target_pool),
            "target_test": counts_by_class(target_test),
        },
        "artifacts": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
    }
    atomic_write_json(output_dir / "task_metadata.json", metadata)
    return metadata


def all_directed_pairs(domains: Iterable[str] = CANONICAL_DOMAINS) -> list[tuple[str, str]]:
    domains = [normalize_domain_name(value) for value in domains]
    return [(source, target) for source in domains for target in domains if source != target]
