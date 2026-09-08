"""Shared read/write helpers for the flat YAML dicts persisted under ``~/.opendatasci``."""

from pathlib import Path
from typing import Any

import yaml


def _load_yaml_dict(path: Path) -> dict[str, Any]:
    """Return the YAML mapping at *path*, or ``{}`` if absent/invalid."""
    try:
        data = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_yaml_dict(path: Path, data: dict[str, Any]) -> None:
    """Write *data* to *path*, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=True))
