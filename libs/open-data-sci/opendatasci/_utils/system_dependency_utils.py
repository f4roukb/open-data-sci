"""Generic OS package-manager detection, reusable by any subsystem that needs
to offer installing missing native dependencies.

Pure and dependency-free by design -- unlike ``opendatasci.sandbox.srt``,
nothing here imports ``sandbox_runtime`` or knows what packages a particular
subsystem needs. See ``accelerator_utils.py`` for the same split applied to
accelerator device discovery: detection lives here, backend-specific glue
(there, bwrap's ``--dev-bind`` injection; for dependency installs, the actual
package names and when to offer installing them) stays with its caller.
"""

import shutil
from typing import Sequence

# Linux package managers this module knows how to drive, in preference order,
# paired with the argv fragment (after the manager name) that performs a
# non-interactive install.
_LINUX_PACKAGE_MANAGERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("apt-get", ("install", "-y")),
    ("dnf", ("install", "-y")),
    ("pacman", ("-S", "--noconfirm")),
)


def find_linux_package_manager() -> str | None:
    """Name of the first supported Linux package manager found on PATH, or ``None``."""
    for manager, _ in _LINUX_PACKAGE_MANAGERS:
        if shutil.which(manager) is not None:
            return manager
    return None


def build_linux_install_command(packages: Sequence[str]) -> list[str] | None:
    """argv to install *packages* via whichever supported package manager is on
    PATH, or ``None`` if none is found.

    Always prefixed with ``sudo`` -- every package manager here requires root,
    unlike Homebrew (see :func:`build_macos_brew_install_command`).
    """
    for manager, install_args in _LINUX_PACKAGE_MANAGERS:
        if shutil.which(manager) is not None:
            return ["sudo", manager, *install_args, *packages]
    return None


def build_macos_brew_install_command(packages: Sequence[str]) -> list[str] | None:
    """argv to install *packages* via Homebrew, or ``None`` if brew isn't installed.

    Never prefixed with ``sudo`` -- Homebrew refuses to run as root.
    """
    if shutil.which("brew") is None:
        return None
    return ["brew", "install", *packages]
