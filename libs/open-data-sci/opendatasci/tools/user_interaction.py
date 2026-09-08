"""User interaction tools: ask_user_mcq."""

from typing import Any, override

from langchain_core.tools import BaseTool
from langgraph.types import interrupt
from pydantic import BaseModel, PrivateAttr

from opendatasci.agents.interrupts import InterruptKind
from opendatasci.tools.base import OpenDataSciBaseTool


class AskUserMcqTool(OpenDataSciBaseTool):
    """Ask the user a multiple-choice question when the task cannot proceed without their input.

    Uses LangGraph's ``interrupt()`` mechanism: the graph is paused and its state
    is persisted to the checkpointer until the caller resumes it via
    ``Command(resume=answer)``.

    Identical questions are deduplicated: the first answer is cached per tool
    instance so the agent never asks the user the same MCQ twice.
    """

    class CallArgs(BaseModel):
        question: str
        choice_a: str
        choice_b: str
        choice_c: str
        summary: str
        communication: str

    name: str = "ask_user_mcq"
    description: str = """\
Ask the user a multiple-choice question when the task cannot proceed without their input.

Presents three predefined choices (A, B, C). The user may also type a free-form answer —
treat any response that doesn't match a choice as a custom answer.

# When to use this tool
- When the problem is genuinely underspecified and the right approach depends on
  an unstated user goal.
- When you need the user's input to make an assumption — ask only when correctness cannot be verified by available means.

# When NOT to use this tool
- For technical decisions you can make yourself — do not delegate judgment.
- When a reasonable assumption would unblock the task — ask only if truly blocked.

Args:
    question:      The question to ask.
    choice_a:      Text for option A.
    choice_b:      Text for option B.
    choice_c:      Text for option C.
    summary:       3-4 word status label (e.g. "Asking user a question").
    communication: Brief message to the user about what you're doing
                   (e.g. "I need your input before I can continue.").\
"""
    args_schema: type[BaseModel] = CallArgs

    _cache: dict[tuple[str, str, str, str], str] = PrivateAttr(default_factory=dict)

    @override
    async def _arun(
        self,
        question: str,
        choice_a: str,
        choice_b: str,
        choice_c: str,
        summary: str,
        communication: str,
        **kwargs: Any,
    ) -> str:
        key = (question, choice_a, choice_b, choice_c)
        if key in self._cache:
            return self._cache[key]

        answer: str = interrupt(
            {
                "kind": InterruptKind.INPUT_REQUIRED,
                "question": question,
                "choices": [choice_a, choice_b, choice_c],
            }
        )
        self._cache[key] = answer
        return answer


def create_user_interaction_tools() -> list[BaseTool]:
    """Return user interaction tools that pause the graph to ask the user a question."""
    return [AskUserMcqTool()]
