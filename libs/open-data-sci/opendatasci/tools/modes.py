"""Agentic-mode tools: switching into Plan/Self-Review mode, and exiting them.

Centralises every tool that mutates ``AgentState.is_plan_mode`` /
``is_self_review_mode`` so the mode lifecycle lives in one place. Entry is a
single tool (``switch_agentic_mode``) that picks the target mode via
:class:`AgentMode`; exit stays mode-specific (``exit_plan_mode`` records the
plan, ``exit_self_review_mode`` records the review) since each returns
different data.
"""

from enum import Enum
from typing import Annotated, Any, override

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict

from opendatasci.agents.states import AgentState
from opendatasci.context.base import BaseContextStore
from opendatasci.skills.base import BaseSkillStore
from opendatasci.tools.base import OpenDataSciBaseTool


class AgentMode(str, Enum):
    """Modes ``switch_agentic_mode`` can enter."""

    PLAN = "plan"
    SELF_REVIEW = "self_review"


_EXIT_TOOL_NAME: dict[AgentMode, str] = {
    AgentMode.PLAN: "exit_plan_mode",
    AgentMode.SELF_REVIEW: "exit_self_review_mode",
}


def _active_mode(state: AgentState) -> AgentMode | None:
    if state.is_plan_mode:
        return AgentMode.PLAN
    if state.is_self_review_mode:
        return AgentMode.SELF_REVIEW
    return None


class SwitchAgentModeTool(OpenDataSciBaseTool):
    """Switch into Plan Mode or Self-Review Mode."""

    class CallArgs(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        mode: AgentMode
        summary: str
        communication: str
        state: Annotated[AgentState, InjectedState]
        tool_call_id: Annotated[str, InjectedToolCallId]
        skill: str | None = None

    name: str = "switch_agentic_mode"
    description: str = """\
Switch into Plan Mode or Self-Review Mode. See the "Modes" section of your \
system prompt for when to use each. Call ``exit_plan_mode`` or \
``exit_self_review_mode`` (whichever matches the mode you entered) to return \
to execution.

Args:
    mode:          Which mode to enter: "plan" or "self_review".
    summary:       3-4 word status label (e.g. "Planning next steps").
    communication: Brief message to the user about what you're doing
                   (e.g. "This task has several interdependent steps — let me plan it first.").
    skill:         Optional skill profile to load before reviewing, only used
                   when mode is "self_review" (e.g. "data_science"). Omit to
                   keep the current skill.\
"""
    args_schema: type[BaseModel] = CallArgs

    store: BaseSkillStore
    context_store: BaseContextStore | None = None
    session_id: str | None = None

    @override
    async def _arun(
        self,
        mode: AgentMode,
        summary: str,
        communication: str,
        state: AgentState,
        tool_call_id: str,
        skill: str | None = None,
        **kwargs: Any,
    ) -> Command[AgentState]:
        current_mode = _active_mode(state)
        if current_mode is not None:
            label = current_mode.value.replace("_", "-")
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=(
                                f"Cannot switch mode: already in {label} mode. "
                                f"Call {_EXIT_TOOL_NAME[current_mode]} first, then "
                                "switch_agentic_mode again."
                            ),
                            tool_call_id=tool_call_id,
                        )
                    ]
                }
            )

        if mode is AgentMode.PLAN:
            return self._enter_plan_mode(tool_call_id)
        return self._enter_self_review_mode(skill, tool_call_id)

    def _enter_plan_mode(self, tool_call_id: str) -> Command[AgentState]:
        if self.context_store is None or self.session_id is None:
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=(
                                "Plan mode is unavailable in this session "
                                "(no persistent context store configured)."
                            ),
                            tool_call_id=tool_call_id,
                        )
                    ]
                }
            )
        return Command(
            update={
                "is_plan_mode": True,
                "messages": [
                    ToolMessage(
                        content=(
                            "Plan Mode active. Think through the full task carefully and produce a "
                            "detailed, ordered plan. Call exit_plan_mode once your plan is complete."
                        ),
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )

    def _enter_self_review_mode(self, skill: str | None, tool_call_id: str) -> Command[AgentState]:
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


class ExitPlanModeTool(OpenDataSciBaseTool):
    """Exit Plan Mode and record the completed plan."""

    class CallArgs(BaseModel):
        final_plan: str
        summary: str
        communication: str
        tool_call_id: Annotated[str, InjectedToolCallId]

    name: str = "exit_plan_mode"
    description: str = """\
Exit Plan Mode and record the completed plan.

The plan is persisted and available as context throughout execution. Write each
step as a concise, single-action description; sequence steps so each one's output
feeds naturally into the next.

Args:
    final_plan:    The complete, ordered plan.
    summary:       3-4 word status label (e.g. "Plan ready").
    communication: Brief message to the user about what you're doing
                   (e.g. "Plan complete — starting execution.").\
"""
    args_schema: type[BaseModel] = CallArgs

    context_store: BaseContextStore
    session_id: str

    @override
    async def _arun(
        self, final_plan: str, summary: str, communication: str, tool_call_id: str, **kwargs: Any
    ) -> Command[AgentState]:
        self.context_store.save_plan(self.session_id, final_plan)
        return Command(
            update={
                "is_plan_mode": False,
                "messages": [
                    ToolMessage(
                        content=(
                            "Plan recorded and saved. You are back in execution mode. "
                            "The plan is now part of your context — work through it step by step."
                        ),
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )


class ExitSelfReviewModeTool(OpenDataSciBaseTool):
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
    async def _arun(
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


def create_mode_tools(
    store: BaseSkillStore,
    context_store: BaseContextStore | None = None,
    session_id: str | None = None,
) -> list[BaseTool]:
    """Return the mode-switching tools for a session.

    ``switch_agentic_mode`` and ``exit_self_review_mode`` are always
    available. ``exit_plan_mode`` (and therefore plan mode itself) requires
    both *context_store* and *session_id*, since the plan must be persisted
    somewhere.

    Args:
        store:         Skill store used to resolve the optional ``skill`` argument.
        context_store: Where completed plans are persisted, if available.
        session_id:    Current session id, used as the plan's storage key.
    """
    tools: list[BaseTool] = [
        SwitchAgentModeTool(store=store, context_store=context_store, session_id=session_id),
        ExitSelfReviewModeTool(),
    ]
    if context_store is not None and session_id is not None:
        tools.append(ExitPlanModeTool(context_store=context_store, session_id=session_id))
    return tools
