"""Root conftest — stubs out unavailable native modules before collection.

``sandbox_runtime`` is an Anthropic-supplied native package that requires
OS-level sandbox binaries (``sandbox-exec`` on macOS, ``bubblewrap`` on
Linux).  It is not installed in CI/unit-test environments, so we inject a
minimal stub into ``sys.modules`` here — *before* pytest collects any test
module — to allow the rest of the package to be imported and tested normally.
"""


import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock


def _make_sandbox_runtime_stub() -> ModuleType:
    mod = ModuleType("sandbox_runtime")
    mod.__path__ = []  # mark as a package so `sandbox_runtime.utils.platform` resolves

    config_cls = MagicMock(name="SandboxRuntimeConfig")
    config_cls.return_value = MagicMock()

    manager_cls = MagicMock(name="SandboxManager")
    manager_cls.initialize = AsyncMock()
    manager_cls.reset = AsyncMock()
    manager_cls.wrap_with_sandbox = AsyncMock(return_value=("", "", 0))
    manager_cls.check_dependencies = MagicMock(return_value=True)
    manager_cls.is_supported_platform = MagicMock(return_value=True)

    mod.SandboxRuntimeConfig = config_cls  # type: ignore[attr-defined]
    mod.SandboxManager = manager_cls  # type: ignore[attr-defined]
    return mod


def _make_sandbox_runtime_utils_platform_stub() -> ModuleType:
    mod = ModuleType("sandbox_runtime.utils.platform")
    mod.get_platform = MagicMock(return_value="linux")  # type: ignore[attr-defined]
    return mod


if "sandbox_runtime" not in sys.modules:
    sys.modules["sandbox_runtime"] = _make_sandbox_runtime_stub()
    utils_mod = ModuleType("sandbox_runtime.utils")
    utils_mod.__path__ = []
    sys.modules["sandbox_runtime.utils"] = utils_mod
    sys.modules["sandbox_runtime.utils.platform"] = _make_sandbox_runtime_utils_platform_stub()
