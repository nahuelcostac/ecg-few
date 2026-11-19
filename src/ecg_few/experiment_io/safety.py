"""Protect source data and final results; validate campaign provenance."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

PROTECTED_PATHS = (("outputs", "final_results_tfg"), ("reports", "loocv"), ("data", "raw"))
PRIVATE_FIELDS = {"api_key", "hf_token", "token", "secret", "password", "authorization"}


def validate_output_path(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    parts = tuple(part.casefold() for part in resolved.parts)
    for protected in PROTECTED_PATHS:
        if any(parts[index : index + len(protected)] == protected for index in range(len(parts))):
            raise ValueError(f"Protected output destination: {resolved}")
    roots = [Path.cwd(), *Path.cwd().parents, *resolved.parents]
    for root in roots:
        if (root / "pyproject.toml").is_file() and (root / "AGENTS.md").is_file():
            for protected in PROTECTED_PATHS:
                target = root.joinpath(*protected).resolve()
                if resolved == target or resolved in target.parents:
                    raise ValueError(
                        f"Output destination contains a protected directory: {resolved}"
                    )
    return resolved


def public_config(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): public_config(item)
            for key, item in value.items()
            if str(key).casefold() not in PRIVATE_FIELDS
            and not str(key).casefold().endswith(("_api_key", "_token", "_password", "_secret"))
        }
    if isinstance(value, (list, tuple)):
        return [public_config(item) for item in value]
    if isinstance(value, Path):
        return str(value.resolve())
    return value


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hash() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def dataset_image_hash(manifest: Path) -> str:
    root = manifest.parent.parent.resolve()
    digest = hashlib.sha256()
    with manifest.open(encoding="utf-8", newline="") as handle:
        image_paths = sorted({row["image_path"] for row in csv.DictReader(handle)})
    for relative in image_paths:
        path = (root / relative).resolve()
        if root not in path.parents:
            raise ValueError("Manifest image paths must stay within the dataset directory.")
        digest.update(relative.encode())
        digest.update(file_hash(path).encode())
    return digest.hexdigest()


def prepare_campaign(
    output_root: Path,
    config: Mapping[str, Any],
    *,
    inputs: Mapping[str, Path],
    resources: Mapping[str, str] | None = None,
    resume: bool,
) -> None:
    from ecg_few.experiment_io.files import save_json

    validate_output_path(output_root)
    excluded = {"resume", "output_root", "report_dir", "config", "limit_folds"}
    effective = public_config({key: value for key, value in config.items() if key not in excluded})
    input_hashes = {name: file_hash(path) for name, path in sorted(inputs.items())}
    image_hashes = {
        name: dataset_image_hash(path)
        for name, path in sorted(inputs.items())
        if path.name == "all_labels.csv"
    }
    resource_hashes = {
        name: hashlib.sha256(content.encode()).hexdigest()
        for name, content in sorted((resources or {}).items())
    }
    identity = {
        "config": effective,
        "inputs": input_hashes,
        "dataset_images": image_hashes,
        "resources": resource_hashes,
        "source_sha256": source_hash(),
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, default=str).encode()
    ).hexdigest()
    path = output_root / "run_manifest.json"
    if path.exists() and resume:
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("fingerprint") != fingerprint:
            raise ValueError(
                "Campaign inputs or configuration changed. Choose a new output directory."
            )
        return
    if output_root.exists() and any(
        item.is_file() and item.name not in {".gitkeep", "README.md"}
        for item in output_root.rglob("*")
    ):
        raise FileExistsError(
            "Campaign directory is not empty. Resume it or choose a new directory."
        )
    dependencies = {}
    for package in ("ecg-vlm", "numpy", "scikit-learn", "torch", "transformers", "openai"):
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            continue
    commit = None
    package_root = Path(__file__).resolve().parents[3]
    if (package_root / ".git").exists():
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=package_root,
            capture_output=True,
            text=True,
            check=False,
        )
        commit = result.stdout.strip() if result.returncode == 0 else None
    save_json(
        path,
        {
            **identity,
            "fingerprint": fingerprint,
            "status": "provisional",
            "created_at": datetime.now(UTC).isoformat(),
            "git_commit": commit,
            "python": sys.version,
            "platform": platform.platform(),
            "dependencies": dependencies,
        },
    )
