import importlib.util
from functools import lru_cache


# Packages whose presence signals the ``[host-dl]`` extra is installed.
DEEP_LEARNING_IMPORT_NAMES: tuple[str, ...] = (
    "torch",
    "jax",
    "transformers",
    "sentence_transformers",
)


@lru_cache(maxsize=1)
def is_host_dl_extra_active() -> bool:
    """Whether a ``[deep-learning]`` package is importable in this environment.

    Checked via ``find_spec`` (not a real import), and cached at module scope,
    so this can't itself trigger a slow or side-effectful import of a
    multi-hundred-MB framework just to answer the question.
    """
    global _deep_learning_active_cache
    if _deep_learning_active_cache is None:
        _deep_learning_active_cache = any(
            importlib.util.find_spec(name) is not None for name in DEEP_LEARNING_IMPORT_NAMES
        )
    return _deep_learning_active_cache