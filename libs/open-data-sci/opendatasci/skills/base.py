from abc import ABC, abstractmethod

from opendatasci._utils.pydantic_utils import FrozenStrictBaseModel


class Skill(FrozenStrictBaseModel):
    """Know-how for carrying out a specific task or subtask.

    A skill is where domain *information* lives — methodologies, idioms,
    defaults, conventions for a focused piece of work. It is injected into
    the agent's system prompt when active.

    Attributes:
        name: Unique skill identifier. Either a bare name for a standalone
            skill (e.g. ``"machine_learning"``) or a ``"<domain>::<skill>"``
            qualified name for a skill that belongs to a skill domain (e.g.
            ``"competitive_data_science::reconnaissance"``).
        content: The prompt text injected into the agent's system prompt when
            this skill is active.
    """

    name: str
    content: str


class SkillDomain(FrozenStrictBaseModel):
    """A collection of skills scoped to a broad task domain.

    A skill domain does not itself contain task-execution know-how — it is
    the map of a domain: which skills exist under it and when each applies.
    The agent loads a skill domain to orient itself, then loads the specific
    skill(s) it points to for the actual how-to.

    Attributes:
        name: Unique skill domain identifier (e.g. ``"competitive_data_science"``).
        content: The skill domain prompt text (pointers to the skills it
            covers), injected into the agent's system prompt when this skill
            domain is active.
    """

    name: str
    content: str


class BaseSkillStore(ABC):
    """Registry of named skills and skill domains available to the agent."""

    @abstractmethod
    def load(self, name: str) -> Skill | None:
        """Return the :class:`Skill` for *name*, or ``None`` if not found.

        *name* is either a bare standalone-skill name or a
        ``"<domain>::<skill>"`` qualified name.
        """

    @abstractmethod
    def load_domain(self, name: str) -> SkillDomain | None:
        """Return the :class:`SkillDomain` for *name*, or ``None`` if not found."""

    @abstractmethod
    def list_skills(self) -> dict[str, Skill]:
        """Return all available skills keyed by name.

        Standalone skills are keyed by their bare name; skills that belong to
        a skill domain are keyed by their ``"<domain>::<skill>"`` qualified name.
        """

    @abstractmethod
    def list_domains(self) -> dict[str, SkillDomain]:
        """Return all available skill domains keyed by name."""
