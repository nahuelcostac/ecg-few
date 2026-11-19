"""Load doctor-editable Markdown prompt templates."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path


def load_markdown_prompt(path: str | Path) -> str:
    """Load a Markdown prompt, stripping simple YAML frontmatter if present."""
    value = str(path)
    if value.startswith("prompts/") and not Path(value).is_file():
        value = "package:" + value.removeprefix("prompts/")
    if value.startswith("package:"):
        relative = value.removeprefix("package:")
        if not relative or any(part in {"", ".", ".."} for part in relative.split("/")):
            raise ValueError("Invalid packaged prompt path.")
        text = (
            files("ecg_few.prompts")
            .joinpath("templates", *relative.split("/"))
            .read_text(encoding="utf-8")
        )
    else:
        text = Path(path).read_text(encoding="utf-8")
    text = text.strip()
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            text = parts[2].strip()
    return text


def render_markdown_prompt(path: str | Path, **values: object) -> str:
    """Load and render a Markdown prompt with Python ``str.format`` variables."""
    return load_markdown_prompt(path).format(**values)
