"""LangChain / LangGraph message and state utilities."""

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.types import StateSnapshot


def get_message_text_content(msg: BaseMessage) -> str:
    """Extract plain text from a message, skipping non-text blocks (e.g. thinking)."""
    content = msg.content
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
        elif isinstance(block, str):
            parts.append(block)
    return "\n".join(parts)


def render_turn(messages: list[BaseMessage]) -> str:
    """Render a sequence of messages (a full turn or an ongoing slice) as a readable string.

    Each message type is formatted as follows:

    - ``HumanMessage`` → ``User: <content>``
    - ``AIMessage`` with tool calls → ``[TOOL CALL: <name>]\\n<args>`` (one entry per call)
    - ``AIMessage`` without tool calls → ``Agent: <text>``
    - ``ToolMessage`` → ``[TOOL OUTPUT]\\n<content>``

    Other message types are silently skipped.
    """
    parts: list[str] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            content = content.strip()
            if content:
                parts.append(f"User: {content}")
        elif isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                args_str = str(tc.get("args", {}))
                parts.append(f"[TOOL CALL: {tc['name']}]\n{args_str}")
        elif isinstance(msg, AIMessage):
            text = get_message_text_content(msg).strip()
            if text:
                parts.append(f"Agent: {text}")
        elif isinstance(msg, ToolMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            parts.append(f"[TOOL OUTPUT]\n{content}")
    return "\n\n".join(parts) if parts else "(no messages)"


def prepend_messages(
    history: list[BaseMessage],
    messages: list[BaseMessage],
) -> list[BaseMessage]:
    """Prepend *messages* to *history*, dropping any existing SystemMessages from *history*."""
    non_system = [m for m in history if not isinstance(m, SystemMessage)]
    return messages + non_system


def is_interrupt_state_snapshot(state: StateSnapshot) -> bool:
    """Return True if *state* contains at least one pending LangGraph interrupt."""
    return any(intr for task in state.tasks for intr in task.interrupts)


def is_final_ai_message(msg: BaseMessage) -> bool:
    """Return True if *msg* is an AIMessage with no pending tool calls."""
    return isinstance(msg, AIMessage) and not bool(getattr(msg, "tool_calls", None))


def get_final_ai_message(chat_history: list[BaseMessage]) -> AIMessage:
    """Return the last AIMessage in *chat_history*, or raise ValueError if none exists."""
    for msg in reversed(chat_history):
        if isinstance(msg, AIMessage):
            return msg
    raise ValueError("No AIMessage found in chat history")


def is_ongoing_turn(turn: list[BaseMessage]) -> bool:
    """Return True if *turn* is an active, in-progress ReAct turn.

    A valid ongoing turn starts with a HumanMessage and ends with either an
    AIMessage carrying pending tool calls, a ToolMessage (tool results not yet
    processed by the agent), or an interrupt-reply HumanMessage (the agent
    paused to ask the user a question and has not yet resumed).
    """
    if not turn:
        return False
    if not isinstance(turn[0], HumanMessage):
        return False
    last = turn[-1]
    if isinstance(last, ToolMessage):
        return True
    if isinstance(last, HumanMessage) and last.additional_kwargs.get(
        "is_input_on_interrupt", False
    ):
        return True
    return isinstance(last, AIMessage) and bool(last.tool_calls)
