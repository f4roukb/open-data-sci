"""Abstract mixins."""

from abc import ABC, abstractmethod


class LLMDigestibleMixin(ABC):
    """Mixin for classes whose instances can be rendered into LLM-facing message content."""

    @abstractmethod
    def to_content(self) -> str:
        """Render this object as plain text suitable for inclusion in an LLM message."""
        ...


class RenderableMessageMixin[MixinSubtype](ABC):
    """Mixin for message types that know how to render themselves for LLM consumption."""

    @abstractmethod
    def render(self) -> MixinSubtype:
        """Return an LLM-ready copy of this message."""
        ...
