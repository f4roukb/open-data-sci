"""Persisted provider credentials collected through the onboarding overlay.

Stores plain field->value pairs (API keys, regions, endpoints, ...) so that
once a user supplies a missing value through the TUI onboarding overlay,
subsequent launches don't ask again. Lowest-precedence source: CLI flags and
environment variables (including ``.env``) always override it.
"""

from pathlib import Path
from typing import Any

from opendatasci._tui.config._yaml_store import _load_yaml_dict, _save_yaml_dict

SECRETS_PATH = Path.home() / ".opendatasci" / "secrets" / "api.yaml"


def load_secrets() -> dict[str, Any]:
    """Return the persisted provider credentials, or ``{}`` if absent/invalid."""
    return _load_yaml_dict(SECRETS_PATH)


def save_secret_value(field: str, value: Any) -> None:
    """Persist a single ``field: value`` pair, preserving any existing entries."""
    data = load_secrets()
    data[field] = value
    _save_yaml_dict(SECRETS_PATH, data)
