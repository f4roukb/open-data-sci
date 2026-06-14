"""Workspace — abstractions and local implementation."""

from opendatasci.workspace.base import BaseWorkspace
from opendatasci.workspace.local import LocalWorkspace

__all__ = [
    "BaseWorkspace",
    "LocalWorkspace",
]
