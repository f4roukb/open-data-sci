"""Self-review mode tools: enter_self_review_mode and exit_self_review_mode."""

import logging
from typing import Annotated, Any, override

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict

from opendatasci.agents.states import AgentState
from opendatasci.skills.base import BaseSkillStore
from opendatasci.tools.base import OpenDataSciSyncTool

logger = logging.getLogger(__name__)


class EnterSelfReviewModeTool(OpenDataSciSyncTool):
    """Enter Self-Review Mode to critically audit your work before continuing."""

    class CallArgs(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        summary: str
        communication: str
        state: Annotated[AgentState, InjectedState]
        tool_call_id: Annotated[str, InjectedToolCallId]
        skill: str | None = None

    name: str = "enter_self_review_mode"
    description: str = """\
Enter Self-Review Mode to critically audit your work before continuing.

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
    summary:       3-4 word status label (e.g. "Reviewing analysis so far").
    communication: Brief message to the user about what you're doing
                   (e.g. "Let me review my work before continuing.").
    skill:         Optional skill profile to load before reviewing
                   (e.g. ``"data_science"``). Omit to keep the current skill.\
"""
    args_schema: type[BaseModel] = CallArgs

    store: BaseSkillStore

    @override
    def _run(
        self,
        summary: str,
        communication: str,
        state: AgentState,
        tool_call_id: str,
        skill: str | None = None,
        **kwargs: Any,
    ) -> Command[AgentState]:
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
            loaded = self.store.load(skill)
            if loaded is None:
                available = ", ".join(sorted(self.store.list_skills()))
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


class ExitSelfReviewModeTool(OpenDataSciSyncTool):
    """Exit Self-Review Mode and record the review findings."""

    class CallArgs(BaseModel):
        review: str
        summary: str
        communication: str
        tool_call_id: Annotated[str, InjectedToolCallId]

    name: str = "exit_self_review_mode"
    description: str = """\
Exit Self-Review Mode and record the review findings.

Returns to execution mode. If missteps were identified, correct course before proceeding.

# How to use this tool
- Reference concrete results, tool calls, or decisions from the conversation.
- Be specific: name what is wrong (or confirm what is sound) — vague assessments are useless.

Args:
    review:        A clear assessment of whether your work is on the right track.
                   Describe any missteps, incorrect assumptions, or missed steps — or
                   confirm that your progress is sound.
    summary:       3-4 word status label (e.g. "Review complete").
    communication: Brief message to the user about what you're doing
                   (e.g. "Review done — continuing execution.").\
"""
    args_schema: type[BaseModel] = CallArgs

    @override
    def _run(
        self, review: str, summary: str, communication: str, tool_call_id: str, **kwargs: Any
    ) -> Command[AgentState]:
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


def create_critic_tools(
    store: BaseSkillStore,
) -> list[BaseTool]:
    """Return ``enter_self_review_mode`` and ``exit_self_review_mode``.

    Args:
        store: Skill store used to resolve the optional skill argument.
    """
    return [
        EnterSelfReviewModeTool(store=store),
        ExitSelfReviewModeTool(),
    ]
