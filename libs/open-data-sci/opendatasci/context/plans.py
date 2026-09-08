"""Session plan representation."""

from typing import Any

from pydantic import Field

from opendatasci._utils.mixins import LLMDigestibleMixin
from opendatasci._utils.pydantic_utils import MutableStrictBaseModel


class Plan(MutableStrictBaseModel, LLMDigestibleMixin):
    """A session plan, persisted alongside arbitrary metadata."""

    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_content(self) -> str:
        return f"<current_plan>\n{self.content}\n</current_plan>"
