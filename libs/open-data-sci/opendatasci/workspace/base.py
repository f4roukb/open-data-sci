"""Abstract base class for workspace containers."""

from abc import ABC, abstractmethod


class BaseWorkspace(ABC):
    """Abstract workspace container.

    A workspace represents the root of the data the agent operates on.
    Implement this class to support custom storage backends.
    """

    @abstractmethod
    def get_reference(self) -> str:
        """Return the canonical location of this workspace as a string.

        For local workspaces this is the absolute directory path; other
        backends may return a URI or a cloud bucket path.
        """
        ...
