"""Skills tool: load a specialised skill profile into the agent's system prompt."""

from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from opendatasci.agents.states import AgentState
from opendatasci.skills import SKILL_LABELS, Skill
from opendatasci.skills.base import BaseSkillStore


def _label_for(skill: Skill) -> str:
    return SKILL_LABELS.get(skill.name) or skill.name.replace("_", " ").title()


def create_skill_tools(store: BaseSkillStore) -> list[BaseTool]:
    """Return the skill tools bound to *store*."""
    return [_create_skill_tool(store)]


def _create_skill_tool(store: BaseSkillStore) -> BaseTool:
    @tool
    def load_skill(
        skill: str,
        summary: str,
        communication: str,
        state: Annotated[AgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command[AgentState]:
        """Load a specialised skill profile to sharpen domain-specific guidance.

        Only one skill is active at a time; loading a new one replaces the previous.

        # When to use this tool
        - At the start of a domain-specific task to get targeted guidance and best practices.
        - When switching task domains mid-session (e.g. from data wrangling to model training).

        # Available skills
        ``data_science``, ``competitive_data_science``,
        ``competitive_data_science_v2``, ``machine_learning``,
        ``deep_learning``, ``quantitative_analysis``, ``data_science_education``.

        Args:
            skill:         Profile name to load.
            summary:       3-4 word status label (e.g. "Loading data science skill").
            communication: Brief message to the user about what you're doing
                           (e.g. "Let me load the data science skill for this task.").
        """
        active = state.active_skills
        if active and active[0].name == skill:
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content="This skill is already loaded.", tool_call_id=tool_call_id
                        )
                    ]
                }
            )

        loaded = store.load(skill)
        if loaded is None:
            available = ", ".join(sorted(store.list()))
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=f"Unknown skill '{skill}'. Available: {available}",
                            tool_call_id=tool_call_id,
                        )
                    ]
                }
            )

        return Command(
            update={
                "active_skills": [loaded],
                "messages": [
                    ToolMessage(
                        content=f"{_label_for(loaded)} skill loaded.", tool_call_id=tool_call_id
                    )
                ],
            }
        )

    return load_skill
