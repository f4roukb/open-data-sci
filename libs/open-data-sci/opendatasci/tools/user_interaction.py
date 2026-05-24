"""User interaction tools: ask_user_mcq."""

from langchain_core.tools import BaseTool, tool
from langgraph.types import interrupt


def create_user_interaction_tools() -> list[BaseTool]:
    """Return user interaction tools that pause the graph to ask the user a question.

    Uses LangGraph's ``interrupt()`` mechanism: the graph is paused and its state
    is persisted to the checkpointer until the caller resumes it via
    ``Command(resume=answer)``.

    Identical questions are deduplicated: the first answer is cached per tool
    instance so the agent never asks the user the same MCQ twice.
    """
    _cache: dict[tuple[str, str, str, str], str] = {}

    @tool
    def ask_user_mcq(
        question: str,
        choice_a: str,
        choice_b: str,
        choice_c: str,
    ) -> str:
        """Ask the user a multiple-choice question when the task cannot proceed without their input.

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
            question: The question to ask.
            choice_a: Text for option A.
            choice_b: Text for option B.
            choice_c: Text for option C.
        """
        key = (question, choice_a, choice_b, choice_c)
        if key in _cache:
            return _cache[key]

        answer: str = interrupt({"question": question, "choices": [choice_a, choice_b, choice_c]})
        _cache[key] = answer
        return answer

    return [ask_user_mcq]
