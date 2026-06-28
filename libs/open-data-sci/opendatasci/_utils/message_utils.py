"""LangChain message utilities."""

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage


def get_message_text_content(msg: BaseMessage) -> str:
    """Extract plain text from a message, skipping non-text blocks (e.g. thinking)."""
    content = msg.content
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return str(content).strip()
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text" and (stripped := block.get("text", "").strip()):
                parts.append(stripped)
        elif isinstance(block, str):
            if stripped := block.strip():
                parts.append(stripped)
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


def is_ai_message_with_tool_calls(msg: BaseMessage) -> bool:
    return isinstance(msg, AIMessage) and bool(msg.tool_calls)


def is_final_ai_message(msg: BaseMessage) -> bool:
    """Return True if *msg* is an AIMessage with no pending tool calls."""
    return isinstance(msg, AIMessage) and not is_ai_message_with_tool_calls(msg)


def get_final_ai_message(chat_history: list[BaseMessage]) -> AIMessage:
    """Return the last AIMessage in *chat_history*, or raise ValueError if none exists."""
    for msg in reversed(chat_history):
        if isinstance(msg, AIMessage):
            return msg
    raise ValueError("No AIMessage found in chat history")


def get_thoughts(msg: AIMessage) -> str:
    """Return the concatenated thinking content from *msg*, or an empty string if none."""
    content = msg.content
    if isinstance(content, str):
        return ""
    return "\n".join(
        block.get("thinking", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "thinking"
    )
