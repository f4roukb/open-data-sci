"""Persisted user settings collected through the startup wizard and ``/config``.

Stores plain field->value pairs (theme, tips, provider/model selections,
temperature, agent name, worker timeout, ...) so that once a user sets a value
through the wizard or the ``/config`` panel, subsequent launches reuse it
instead of asking or falling back to defaults.
"""

from pathlib import Path
from typing import Any

from opendatasci._tui.config._yaml_store import _load_yaml_dict, _save_yaml_dict

SETTINGS_PATH = Path.home() / ".opendatasci" / "settings" / "global.yaml"


def load_settings() -> dict[str, Any]:
    """Return the persisted settings, or ``{}`` if absent/invalid."""
    return _load_yaml_dict(SETTINGS_PATH)


def save_settings_values(values: dict[str, Any]) -> None:
    """Persist ``field: value`` pairs, preserving any existing entries."""
    data = load_settings()
    data.update(values)
    _save_yaml_dict(SETTINGS_PATH, data)
