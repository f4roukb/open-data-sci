"""Skills tools: load_skill and list_skills.

Two layers of domain knowledge are exposed here:

- A **skill domain** is a collection of skills scoped to a broad task
  domain (e.g. a competition playbook). It does not itself carry
  task-execution know-how — it is the map of the domain, pointing to the
  skills that do.
- A **skill** carries the actual know-how for a specific task or subtask —
  methodologies, idioms, defaults, conventions.

``list_skills`` surfaces what is available; ``load_skill`` loads a skill,
a skill domain, or both into the agent's system prompt.
"""

import json
from typing import Annotated, Any, override

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict

from opendatasci.agents.states import AgentState
from opendatasci.skills import Skill, SkillDomain
from opendatasci.skills.base import BaseSkillStore
from opendatasci.skills.local import SKILL_LABELS
from opendatasci.tools.base import OpenDataSciBaseTool


def _label_for(name: str) -> str:
    if name in SKILL_LABELS:
        return SKILL_LABELS[name]
    tail = name.rsplit("::", 1)[-1]
    return tail.replace("_", " ").title()


class LoadSkillTool(OpenDataSciBaseTool):
    """Load a specific skill and/or a skill domain to inform your work."""

    class CallArgs(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        summary: str
        communication: str
        state: Annotated[AgentState, InjectedState]
        tool_call_id: Annotated[str, InjectedToolCallId]
        skill_name: str | None = None
        skill_domain_name: str | None = None

    name: str = "load_skill"
    description: str = """\
Load a specific skill and/or a skill domain to inform your work.

A skill domain is a collection of skills for a broad task domain —
loading one gives you the map of that domain (which skills exist
under it and when to reach for each) without itself teaching you how
to do the work. A skill carries the actual know-how for a specific
task or subtask.

Call ``list_skills`` first if you are not sure what is available. Only
one skill domain and one skill are active at a time: loading a new
skill overrides the currently loaded skill, and loading a new skill
domain overrides the currently loaded skill domain — each is replaced
independently, so loading one does not clear the other.

# When to use this tool
- At the start of a domain-specific task: load the domain for that
  task, then the specific skill it points you to.
- When you already know the exact skill you need, load it directly
  via ``skill_name`` without a domain.
- When switching task domains mid-session.

Args:
    skill_name:        Name of the skill to load. Bare for a
                       standalone skill (e.g. ``"data_science"``) or
                       qualified for a skill that belongs to a domain
                       (e.g. ``"competitive_data_science::reconnaissance"``).
                       Omit to leave the current skill unchanged.
    skill_domain_name: Name of the skill domain to load
                       (e.g. ``"competitive_data_science"``). Omit to
                       leave the current domain (or lack thereof)
                       unchanged.
    summary:           3-4 word status label (e.g. "Loading data science skill").
    communication:     Brief message to the user about what you're doing
                       (e.g. "Let me load the data science skill for this task.").\
"""
    args_schema: type[BaseModel] = CallArgs

    store: BaseSkillStore

    @override
    async def _arun(
        self,
        summary: str,
        communication: str,
        state: AgentState,
        tool_call_id: str,
        skill_name: str | None = None,
        skill_domain_name: str | None = None,
        **kwargs: Any,
    ) -> Command[AgentState]:
        if skill_domain_name is None and skill_name is None:
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=(
                                "At least one of `skill_name` or `skill_domain_name` must be "
                                "provided. Call `list_skills` to see what is available."
                            ),
                            tool_call_id=tool_call_id,
                        )
                    ]
                }
            )

        state_update: dict[str, object] = {}
        confirmations: list[str] = []

        if skill_domain_name is not None:
            current_skill_domains = state.active_skill_domains
            current_skill_domain = current_skill_domains[0] if current_skill_domains else None
            if current_skill_domain is not None and current_skill_domain.name == skill_domain_name:
                confirmations.append(f"Skill domain '{skill_domain_name}' is already loaded.")
            else:
                skill_domain: SkillDomain | None = self.store.load_domain(skill_domain_name)
                if skill_domain is None:
                    available = ", ".join(sorted(self.store.list_domains())) or "(none)"
                    return Command(
                        update={
                            "messages": [
                                ToolMessage(
                                    content=(
                                        f"Unknown skill domain '{skill_domain_name}'. "
                                        f"Available skill domains: {available}"
                                    ),
                                    tool_call_id=tool_call_id,
                                )
                            ]
                        }
                    )
                state_update["active_skill_domains"] = [skill_domain]
                confirmations.append(f"{_label_for(skill_domain.name)} skill domain loaded.")

        if skill_name is not None:
            current_skills = state.active_skills
            if current_skills and current_skills[0].name == skill_name:
                confirmations.append(f"Skill '{skill_name}' is already loaded.")
            else:
                loaded: Skill | None = self.store.load(skill_name)
                if loaded is None:
                    available = ", ".join(sorted(self.store.list_skills())) or "(none)"
                    return Command(
                        update={
                            "messages": [
                                ToolMessage(
                                    content=(
                                        f"Unknown skill '{skill_name}'. Available skills: {available}"
                                    ),
                                    tool_call_id=tool_call_id,
                                )
                            ]
                        }
                    )
                state_update["active_skills"] = [loaded]
                confirmations.append(f"{_label_for(loaded.name)} skill loaded.")

        state_update["messages"] = [
            ToolMessage(content=" ".join(confirmations), tool_call_id=tool_call_id)
        ]
        return Command(update=state_update)


class ListSkillsTool(OpenDataSciBaseTool):
    """List the skill domains and standalone skills available to load."""

    class CallArgs(BaseModel):
        summary: str
        communication: str

    name: str = "list_skills"
    description: str = """\
List the skill domains and standalone skills available to load.

A skill domain is a collection of skills for a broad task domain; a
standalone skill carries know-how for a specific task on its own.
Load a skill domain via ``load_skill(skill_domain_name=...)`` to see
the specific skills it points to, or load a standalone skill directly
via ``load_skill(skill_name=...)``.

Returns a JSON object with ``domains`` (skill domain names) and
``standalone_skills`` (skill names not tied to any domain).

Args:
    summary:       3-4 word status label (e.g. "Checking available skills").
    communication: Brief message to the user about what you're doing
                   (e.g. "Let me check what skills are available.").\
"""
    args_schema: type[BaseModel] = CallArgs

    store: BaseSkillStore

    @override
    async def _arun(self, summary: str, communication: str, **kwargs: Any) -> str:
        domains = sorted(self.store.list_domains())
        standalone_skills = sorted(name for name in self.store.list_skills() if "::" not in name)
        return json.dumps({"domains": domains, "standalone_skills": standalone_skills})


def create_skill_tools(store: BaseSkillStore) -> list[BaseTool]:
    """Return the ``load_skill`` and ``list_skills`` tools bound to *store*."""
    return [LoadSkillTool(store=store), ListSkillsTool(store=store)]
