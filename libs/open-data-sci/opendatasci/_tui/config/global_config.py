"""Persisted global config for values collected through the onboarding overlay.

Stores plain field->value pairs (API keys, regions, endpoints, ...) so that
once a user supplies a missing value through the TUI onboarding overlay,
subsequent launches don't ask again. Lowest-precedence source: CLI flags and
environment variables (including ``.env``) always override it.
"""

from pathlib import Path
from typing import Any

import yaml

GLOBAL_CONFIG_PATH = Path.home() / ".opendatasci" / "config.yaml"


def load_global_config() -> dict[str, Any]:
    """Return the persisted global config, or ``{}`` if absent/invalid."""
    try:
        data = yaml.safe_load(GLOBAL_CONFIG_PATH.read_text())
    except (OSError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


def save_global_config_value(field: str, value: Any) -> None:
    """Persist a single ``field: value`` pair, preserving any existing entries."""
    data = load_global_config()
    data[field] = value
    GLOBAL_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    GLOBAL_CONFIG_PATH.write_text(yaml.safe_dump(data, sort_keys=True))
