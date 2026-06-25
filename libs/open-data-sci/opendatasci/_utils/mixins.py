"""Mixin for objects that can be rendered as LLM-digestible text content."""

from abc import ABC, abstractmethod

__all__ = ["LLMDigestibleMixin"]


class LLMDigestibleMixin(ABC):
    """Mixin for classes whose instances can be rendered into LLM-facing message content."""

    @abstractmethod
    def to_content(self) -> str:
        """Render this object as plain text suitable for inclusion in an LLM message."""
        ...
