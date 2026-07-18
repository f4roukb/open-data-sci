import re

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def camel_to_snake(name: str) -> str:
    """Convert a camelCase (or already-snake_case) string to snake_case."""
    return _CAMEL_BOUNDARY.sub("_", name).lower()
