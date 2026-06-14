from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Skill:
    """A named prompt extension that specialises the agent for a domain.

    Attributes:
        name: Unique skill identifier (e.g. ``"machine_learning"``).
        content: The prompt text injected into the agent's system prompt when
            this skill is active.
    """

    name: str
    content: str


class BaseSkillStore(ABC):
    """Registry of named skills available to the agent."""

    @abstractmethod
    def load(self, name: str) -> Skill | None:
        """Return the :class:`Skill` for *name*, or ``None`` if not found."""

    @abstractmethod
    def list(self) -> dict[str, Skill]:
        """Return all available skills keyed by name."""
