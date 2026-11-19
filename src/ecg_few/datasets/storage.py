"""Manage generated datasets without removing their tracked layout."""

from __future__ import annotations

from pathlib import Path

from ecg_few.experiment_io import validate_output_path

STRUCTURE_FILES = {".gitkeep", ".gitignore", "README.md"}


def reset_output_dir(outdir: Path, *, overwrite: bool) -> None:
    if outdir.is_symlink():
        raise ValueError("Dataset output must not be a symbolic link.")
    target = validate_output_path(outdir)
    if target.exists() and not target.is_dir():
        raise NotADirectoryError(target)
    entries = list(target.rglob("*")) if target.exists() else []
    if any(path.is_symlink() for path in entries):
        raise ValueError("Dataset output must not contain symbolic links.")
    allowed = {"images", "labels", "vlm", "examples", "metadata", "metadata.jsonl", "metadata.json"}
    if target.exists() and any(
        path.name not in STRUCTURE_FILES | allowed for path in target.iterdir()
    ):
        raise ValueError("Dataset output contains files outside the generated dataset layout.")
    generated = [path for path in entries if path.is_file() and path.name not in STRUCTURE_FILES]
    if generated and not overwrite:
        raise FileExistsError(
            f"Dataset already contains generated files: {target}. Use --overwrite."
        )
    for path in generated:
        path.unlink()
    for path in sorted(
        (path for path in entries if path.is_dir()), key=lambda p: len(p.parts), reverse=True
    ):
        if not any(path.iterdir()):
            path.rmdir()
    target.mkdir(parents=True, exist_ok=True)
