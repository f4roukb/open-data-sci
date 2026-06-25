"""Abstract base classes for the agent's context stores."""

from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Self

from opendatasci.context.plans import Plan

__all__ = ["BaseContextStore", "Plan"]


class BaseContextStore(ABC):
    """Context store for dataset notes, profile cards, and session plans.

    Dataset notes and profile cards persist across agent sessions and are keyed
    by dataset path.  Plans are scoped to a session and keyed by ``session_id``.
    """

    @abstractmethod
    def session(self) -> AbstractAsyncContextManager[Self]:
        """Return an async context manager that manages this store's lifecycle."""

    @property
    @abstractmethod
    def root(self) -> Path:
        """Return the root path of the context store (e.g. the ``.opendatasci`` directory)."""

    @abstractmethod
    async def read_dataset_info(self, dataset_path: str) -> str:
        """Return combined dataset info: profile card (if any) + session notes."""

    @abstractmethod
    async def update_dataset_info(
        self,
        dataset_path: str,
        update: str,
        merge: bool = True,
    ) -> str:
        """Persist dataset notes and return the path to the stored notes file."""

    @abstractmethod
    async def get_profile_info(self, dataset_path: str) -> tuple[str, str, str | None]:
        """Return ``(resolved_path, content_hash, existing_profile_or_None)`` for *dataset_path*."""

    @abstractmethod
    def save_dataset_profile(self, hash_hex: str, content: str) -> None:
        """Persist a completed profile card for *hash_hex*."""

    @abstractmethod
    def get_current_plan(self, session_id: str) -> Plan | None:
        """Return the most recent plan for *session_id*, or ``None``."""

    @abstractmethod
    def save_plan(self, session_id: str, content: str) -> None:
        """Persist a new plan with *content* for *session_id*."""
