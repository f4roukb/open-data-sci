import importlib.util
from functools import lru_cache

# Packages whose presence signals the ``[host-dl]`` extra is installed.
HOST_DL_IMPORT_NAMES: tuple[str, ...] = (
    "torch",
    "jax",
    "transformers",
    "sentence_transformers",
)


@lru_cache(maxsize=1)
def is_host_dl_extra_active() -> bool:
    """Whether a ``[host-dl]`` package is importable in this environment.

    Checked via ``find_spec`` (not a real import), and cached, so this can't
    itself trigger a slow or side-effectful import of a multi-hundred-MB
    framework just to answer the question.
    """
    return any(importlib.util.find_spec(name) is not None for name in HOST_DL_IMPORT_NAMES)
