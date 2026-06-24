"""Session plan representation."""

from dataclasses import dataclass, field

from opendatasci._utils.mixins import LLMDigestibleMixin

__all__ = ["Plan"]


@dataclass
class Plan(LLMDigestibleMixin):
    """A session plan, persisted alongside arbitrary metadata."""

    content: str
    metadata: dict = field(default_factory=dict)

    def to_content(self) -> str:
        return f"<current_plan>\n{self.content}\n</current_plan>"
