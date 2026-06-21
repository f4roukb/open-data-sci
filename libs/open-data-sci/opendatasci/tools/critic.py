"""Self-review mode tools: enter_self_review_mode and exit_self_review_mode."""

import logging
from typing import Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from opendatasci.agents.states import AgentState
from opendatasci.skills.base import BaseSkillStore

logger = logging.getLogger(__name__)


def create_critic_tools(
    store: BaseSkillStore,
) -> list[BaseTool]:
    """Return ``enter_self_review_mode`` and ``exit_self_review_mode``.

    Args:
        store: Skill store used to resolve the optional skill argument.
    """
    return [
        _create_enter_tool(store),
        _create_exit_tool(),
    ]


def _create_enter_tool(store: BaseSkillStore) -> BaseTool:
    @tool
    def enter_self_review_mode(
        state: Annotated[AgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        skill: str | None = None,
    ) -> Command[AgentState]:
        """Enter Self-Review Mode to critically audit your work before continuing.

        In Self-Review Mode only read-only tools are available. Call
        ``exit_self_review_mode`` with your full review to return to execution.

        # When to use this tool
        - After a complex multi-step analysis to verify that your methodology is sound and your key results were obtained correctly.
        - When results look surprising or inconsistent with expectations.
        - Before a consequential decision that depends heavily on prior work.

        # When NOT to use this tool
        - While Plan Mode is active — exit plan mode first.
        - For routine single-step work where there is nothing meaningful to review.

        Args:
            skill: Optional skill profile to load before reviewing
                   (e.g. ``"data_science"``). Omit to keep the current skill.
        """
        if state.is_plan_mode:
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=(
                                "Cannot enter self-review mode while plan mode is active. "
                                "Exit plan mode first, then call enter_self_review_mode."
                            ),
                            tool_call_id=tool_call_id,
                        )
                    ]
                }
            )

        state_update: dict[str, Any] = {"is_self_review_mode": True}
        if skill is not None:
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
            state_update["active_skills"] = [loaded]

        state_update["messages"] = [
            ToolMessage(
                content=(
                    "Self-review mode active. Review the entire conversation, all results, "
                    "plans, dataset notes, and artefacts produced so far, then assess whether "
                    "the analysis is on the right track. "
                    "Call exit_self_review_mode once your review is complete."
                ),
                tool_call_id=tool_call_id,
            )
        ]
        return Command(update=state_update)

    return enter_self_review_mode


def _create_exit_tool() -> BaseTool:
    @tool
    def exit_self_review_mode(
        review: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command[AgentState]:
        """Exit Self-Review Mode and record the review findings.

        Returns to execution mode. If missteps were identified, correct course before proceeding.

        # How to use this tool
        - Reference concrete results, tool calls, or decisions from the conversation.
        - Be specific: name what is wrong (or confirm what is sound) — vague assessments are useless.

        Args:
            review: A clear assessment of whether your work is on the right track.
                    Describe any missteps, incorrect assumptions, or missed steps — or
                    confirm that your progress is sound.
        """
        content = (
            f"Self-review complete. Review recorded:\n\n{review}\n\n"
            "You are back in execution mode. "
            "If missteps were identified, correct course before proceeding."
        )
        return Command(
            update={
                "is_self_review_mode": False,
                "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)],
            }
        )

    return exit_self_review_mode
