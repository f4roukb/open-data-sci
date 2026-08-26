import re
from typing import Any

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def camel_to_snake(name: str) -> str:
    """Convert a camelCase (or already-snake_case) string to snake_case."""
    return _CAMEL_BOUNDARY.sub("_", name).lower()


def camel_to_snake_keys(value: Any) -> Any:
    """Recursively rewrite camelCase dict keys to snake_case.

    Walks dicts and lists, converting every dict key (via
    :func:`camel_to_snake`) and leaving other values untouched. Keys
    already in snake_case pass through unchanged.
    """
    if isinstance(value, dict):
        return {
            camel_to_snake(k) if isinstance(k, str) else k: camel_to_snake_keys(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [camel_to_snake_keys(v) for v in value]
    return value
