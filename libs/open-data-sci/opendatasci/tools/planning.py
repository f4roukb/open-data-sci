"""Plan-mode tools: enter_plan_mode and exit_plan_mode."""

from typing import Annotated, Any, override

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command
from pydantic import BaseModel

from opendatasci.agents.states import AgentState
from opendatasci.context.base import BaseContextStore
from opendatasci.tools.base import OpenDataSciSyncTool


class EnterPlanModeTool(OpenDataSciSyncTool):
    """Enter Plan Mode to decompose a complex task before executing it."""

    class CallArgs(BaseModel):
        summary: str
        communication: str
        tool_call_id: Annotated[str, InjectedToolCallId]

    name: str = "enter_plan_mode"
    description: str = """\
Enter Plan Mode to decompose a complex task before executing it.

In Plan Mode you can think through the full problem and produce an ordered,
actionable plan. Call ``exit_plan_mode`` with the completed plan to return to execution.

# When to use this tool
- For tasks with more than two or three interdependent steps — e.g. building a
  full ML pipeline, multi-stage analysis, or anything where step ordering matters.

# When NOT to use this tool
- For simple tasks — the overhead is wasteful.

Args:
    communication: Brief message to the user about what you're doing
                   (e.g. "This task has several interdependent steps — let me plan it first.").\
"""
    args_schema: type[BaseModel] = CallArgs

    @override
    def _run(
        self, summary: str, communication: str, tool_call_id: str, **kwargs: Any
    ) -> Command[AgentState]:
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


class ExitPlanModeTool(OpenDataSciSyncTool):
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
    def _run(
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


def create_planning_tools(
    context_store: BaseContextStore,
    session_id: str,
) -> list[BaseTool]:
    """Return the ``enter_plan_mode`` and ``exit_plan_mode`` tools for *session_id*."""
    return [
        EnterPlanModeTool(),
        ExitPlanModeTool(context_store=context_store, session_id=session_id),
    ]
