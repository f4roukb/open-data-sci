"""Plan-mode tools: enter_plan_mode and exit_plan_mode."""

from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command

from opendatasci.agents.states import AgentState
from opendatasci.context.base import BaseContextStore


def create_planning_tools(
    context_store: BaseContextStore,
    session_id: str,
) -> list[BaseTool]:
    """Return the ``enter_plan_mode`` and ``exit_plan_mode`` tools for *session_id*."""

    @tool
    def enter_plan_mode(
        communication: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command[AgentState]:
        """Enter Plan Mode to decompose a complex task before executing it.

        In Plan Mode you can think through the full problem and produce an ordered,
        actionable plan. Call ``exit_plan_mode`` with the completed plan to return to execution.

        # When to use this tool
        - For tasks with more than two or three interdependent steps — e.g. building a
          full ML pipeline, multi-stage analysis, or anything where step ordering matters.

        # When NOT to use this tool
        - For simple tasks — the overhead is wasteful.

        Args:
            communication: Brief message to the user about what you're doing
                           (e.g. "This task has several interdependent steps — let me plan it first.").
        """
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

    @tool
    def exit_plan_mode(
        final_plan: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command[AgentState]:
        """Exit Plan Mode and record the completed plan.

        The plan is persisted and available as context throughout execution. Write each
        step as a concise, single-action description; sequence steps so each one's output
        feeds naturally into the next.

        Args:
            final_plan: The complete, ordered plan.
        """
        context_store.save_plan(session_id, final_plan)
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

    return [enter_plan_mode, exit_plan_mode]
