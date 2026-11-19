"""Load experiment defaults from TOML while preserving CLI overrides."""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path


def parse_config_args(
    parser: argparse.ArgumentParser, argv: list[str] | None = None
) -> argparse.Namespace:
    parser.add_argument("--config", type=Path, help="TOML file containing an [experiment] table.")
    preliminary, _ = parser.parse_known_args(argv)
    if preliminary.config is not None:
        with preliminary.config.open("rb") as handle:
            document = tomllib.load(handle)
        if set(document) != {"experiment"} or not isinstance(document["experiment"], dict):
            parser.error("Configuration must contain a single [experiment] table.")
        actions = {action.dest: action for action in parser._actions}
        defaults = {}
        for key, value in document["experiment"].items():
            if key not in actions or key in {"help", "config"}:
                parser.error(f"Unknown configuration field: {key}")
            action = actions[key]
            if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
                if not isinstance(value, bool):
                    parser.error(f"Configuration field {key} must be a boolean.")
            else:
                if isinstance(value, list):
                    value = ",".join(str(item) for item in value)
                try:
                    value = action.type(value) if action.type else value
                except (TypeError, ValueError) as exc:
                    parser.error(f"Invalid configuration field {key}: {exc}")
                if action.choices is not None and value not in action.choices:
                    parser.error(f"Invalid configuration value for {key}: {value}")
            defaults[key] = value
        parser.set_defaults(**defaults)
    return parser.parse_args(argv)
