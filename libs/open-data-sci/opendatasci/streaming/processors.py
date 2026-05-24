import logging
import re
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from opendatasci._utils.langchain_utils import get_message_text_content
from opendatasci.streaming.events import (
    AgentStreamEvent,
    MessageEvent,
    ReasoningEvent,
    SubagentEvent,
    TokenEvent,
    ToolCallEvent,
    ToolCommunicationEvent,
    ToolResultEvent,
    UsageEvent,
    WorkerDoneEvent,
)

logger = logging.getLogger(__name__)


SUBAGENT_TAG: str = "opendatasci:subagent"


# Regex to extract the `communication` value from a partial tool-call JSON string.
_COMM_RE = re.compile(r'"communication"\s*:\s*"((?:[^"\\]|\\.)*)"')


class AgentTurnStreamProcessor:
    """Converts raw LangGraph stream events into AgentStreamEvent objects.

    Maintains per-stream state (current text buffer, pending tool calls, etc.).
    Call ``process_event`` for each raw event; it returns zero or more
    ``AgentStreamEvent`` objects to yield.  ``MessageEvent`` events carry
    completed ``BaseMessage`` objects and are intended for callers that own
    conversation-history accumulation.
    """

    def __init__(self) -> None:
        # Per-turn translation state
        self._has_streamed_tokens: bool = False
        self._last_communication: dict[str, str] = {}
        self._pending_tool_calls: list[dict[str, Any]] = []
        # Per-model-call state for incremental usage estimates
        self._stream_input_tokens: int | None = None
        self._stream_output_chars: int = 0

    def process_event(self, event: dict[str, Any]) -> list[AgentStreamEvent]:
        """Process one raw graph event and return any StreamEvents to emit."""
        # Drop events that bubbled up from a sub-agent (ParallelWorkerAgent) so the
        # TUI sees only the main agent's narration, tool calls, and the
        # high-level worker progress signals. Worker activity is surfaced via
        # ``SubagentEvent``/``WorkerDoneEvent`` events instead.
        if SUBAGENT_TAG in (event.get("tags") or ()):
            return []
        kind = event.get("event", "")
        if kind == "on_chat_model_stream":
            return self._handle_stream(event)
        if kind == "on_chain_end" and event.get("name") == "agent":
            return self._handle_chain_end(event)
        if kind == "on_tool_end":
            return self._handle_tool_end(event)
        if kind == "on_chat_model_end":
            return self._handle_model_end(event)
        if kind == "on_custom_event" and event.get("name") == "worker_event":
            return self._handle_worker_event(event)
        return []

    def _handle_stream(self, event: dict[str, Any]) -> list[AgentStreamEvent]:
        out: list[AgentStreamEvent] = []
        chunk = event.get("data", {}).get("chunk")
        if not chunk:
            return out
        out.extend(self._extract_content_events(chunk.content))
        self._accumulate_tool_call_chunks(getattr(chunk, "tool_call_chunks", []), out)
        self._update_stream_usage(chunk, out)
        return out

    def _extract_content_events(self, content: Any) -> list[AgentStreamEvent]:
        """Convert chunk content (str or list of blocks) to TokenEvent/ReasoningEvent objects."""
        out: list[AgentStreamEvent] = []
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    btype = block.get("type", "")
                    if btype == "thinking":
                        token = block.get("thinking", "")
                        if token:
                            out.append(ReasoningEvent(content=token))
                    elif btype == "reasoning_content":
                        # Bedrock ConverseStream format
                        token = block.get("reasoning_content", {}).get("text", "")
                        if token:
                            out.append(ReasoningEvent(content=token))
                    elif btype == "text":
                        token = block.get("text", "")
                        if token:
                            self._has_streamed_tokens = True
                            out.append(TokenEvent(content=token))
                elif isinstance(block, str) and block:
                    self._has_streamed_tokens = True
                    out.append(TokenEvent(content=block))
                else:
                    logger.error(
                        "Unhandled stream block type: %s, value: %r",
                        type(block).__name__,
                        block,
                    )
        elif isinstance(content, str) and content:
            self._has_streamed_tokens = True
            out.append(TokenEvent(content=content))
        return out

    def _resolve_tool_call_target(self, chunk_index: int | None) -> dict[str, Any] | None:
        """Return the pending tool call to update for the given chunk index."""
        if not self._pending_tool_calls:
            return None
        if chunk_index is not None:
            return next(
                (tc for tc in self._pending_tool_calls if tc.get("index") == chunk_index),
                self._pending_tool_calls[-1],
            )
        return self._pending_tool_calls[-1]

    def _accumulate_tool_call_chunks(
        self, tc_chunks: list[Any], out: list[AgentStreamEvent]
    ) -> None:
        """Accumulate tool-call arg chunks and emit ToolCommunicationEvent objects."""
        for tc_chunk in tc_chunks:
            chunk_index = tc_chunk.get("index")
            if tc_chunk.get("name"):
                self._pending_tool_calls.append(
                    {
                        "name": tc_chunk.get("name"),
                        "args": tc_chunk.get("args") or "",
                        "id": tc_chunk.get("id"),
                        "index": chunk_index,
                    }
                )
            elif tc_chunk.get("args"):
                # Route args to the tool call identified by index (parallel tool calls).
                # Fall back to the last tool call when no index is present.
                target = self._resolve_tool_call_target(chunk_index)
                if target is not None:
                    target["args"] += tc_chunk.get("args") or ""

            # Inspect the same tool call that was just updated so that each
            # parallel tool call's communication is emitted independently.
            current_tc = self._resolve_tool_call_target(chunk_index)
            if current_tc is None:
                continue
            args = current_tc["args"]
            tc_id = current_tc.get("id") or ""
            m = _COMM_RE.search(args) if '"communication"' in args else None
            if m:
                comm = m.group(1)
                if comm != self._last_communication.get(tc_id, ""):
                    self._last_communication[tc_id] = comm
                    out.append(
                        ToolCommunicationEvent(
                            content=comm,
                            tool_call_id=tc_id,
                            tool_name=current_tc.get("name", ""),
                        )
                    )

    def _update_stream_usage(self, chunk: Any, out: list[AgentStreamEvent]) -> None:
        """Capture input tokens once and emit an incremental usage estimate per text chunk."""
        # Anthropic includes input_tokens in the initial message_start chunk.
        usage_meta = getattr(chunk, "usage_metadata", None)
        if isinstance(usage_meta, dict) and self._stream_input_tokens is None:
            in_tok = usage_meta.get("input_tokens")
            if in_tok:
                self._stream_input_tokens = int(in_tok)

        chars_this_call = sum(len(ev.content) for ev in out if isinstance(ev, TokenEvent))
        if chars_this_call > 0 and self._stream_input_tokens is not None:
            self._stream_output_chars += chars_this_call
            out.append(
                UsageEvent(
                    input_tokens=self._stream_input_tokens,
                    output_tokens=max(1, self._stream_output_chars // 4),
                )
            )

    def _handle_chain_end(self, event: dict[str, Any]) -> list[AgentStreamEvent]:
        out: list[AgentStreamEvent] = []
        output_messages = event.get("data", {}).get("output", {}).get("messages", [])

        for msg in output_messages:
            if not isinstance(msg, AIMessage):
                continue
            out.append(MessageEvent(message=msg))

            if getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    out.extend(self._handle_tool_call(tc))
            else:
                if not self._has_streamed_tokens:
                    # Streaming didn't capture any tokens for this step.
                    # This can happen when ChatAnthropic._agenerate falls
                    # through to the non-streaming API call (streaming=False
                    # and no _StreamingCallbackHandler in the run manager).
                    # The complete text is always on the AIMessage, so extract
                    # it as a fallback so the response is never silently dropped.
                    logger.warning(
                        "_handle_chain_end: no tokens were streamed but AIMessage "
                        "has content — streaming tokens were not captured for this "
                        "step. Falling back to msg.content."
                    )
                    fallback_text = get_message_text_content(msg).strip()
                    if fallback_text:
                        out.append(TokenEvent(content=fallback_text))

        # Reset per-message accumulators.
        self._has_streamed_tokens = False
        self._last_communication = {}
        self._pending_tool_calls = []
        self._stream_input_tokens = None
        self._stream_output_chars = 0
        return out

    def _handle_tool_call(self, tc: Any) -> list[AgentStreamEvent]:
        from opendatasci.tools import ToolName  # local import breaks circular dependency

        name = tc["name"]
        args = tc["args"] if isinstance(tc["args"], dict) else {}
        events: list[AgentStreamEvent] = []

        # Guarantee tool_communication is emitted before tool_call using the
        # complete, finalised args.  The streaming path (_handle_stream) emits
        # it incrementally as chunks arrive, but it may miss the value when the
        # model does not stream tool-call args or when the closing quote of the
        # communication string only arrives in the last chunk.  Using
        # _last_communication as a guard avoids emitting a duplicate when the
        # streaming path already captured the value.
        tc_id = tc.get("id") or ""
        comm = args.get("communication", "")
        if comm and comm != self._last_communication.get(tc_id, ""):
            self._last_communication[tc_id] = comm
            events.append(
                ToolCommunicationEvent(
                    content=comm,
                    tool_call_id=tc_id,
                    tool_name=name,
                )
            )

        # spawn_workers carries worker_summaries instead of a plain summary.
        if name == ToolName.SPAWN_WORKERS:
            tasks_arg = args.get("subtasks", [])
            worker_summaries = [
                t.get("summary", f"ParallelWorkerAgent {i + 1}")
                if isinstance(t, dict)
                else f"ParallelWorkerAgent {i + 1}"
                for i, t in enumerate(tasks_arg)
            ]
            events.append(
                ToolCallEvent(
                    content=str(tc["args"]),
                    # ``ToolName`` is a (str, Enum), and ``str(member)`` returns
                    # ``"ToolName.SPAWN_WORKERS"`` rather than ``"spawn_workers"``.
                    # Storing ``name`` here keeps this branch consistent with the
                    # generic branch below — the presenter does
                    # ``str(event.tool)`` before comparing, so storing the enum
                    # here breaks its spawn_workers detection.
                    tool=name,
                    tool_call_id=tc.get("id"),
                    worker_summaries=worker_summaries,
                )
            )
            return events

        events.append(
            ToolCallEvent(
                content=str(tc["args"]),
                tool=name,
                tool_call_id=tc.get("id"),
                summary=args.get("summary", ""),
            )
        )
        return events

    def _handle_worker_event(self, event: dict[str, Any]) -> list[AgentStreamEvent]:
        data = event.get("data", {})
        event_type = data.get("event_type", "")
        worker_idx = data.get("worker_idx")
        if event_type == "worker_done":
            return [
                WorkerDoneEvent(
                    worker_idx=worker_idx,
                    success=data.get("success", True),
                )
            ]
        return [
            SubagentEvent(
                content=data.get("content", ""),
                worker_idx=worker_idx,
                event_type=event_type,
                success=data.get("success", True),
                summary=data.get("summary", ""),
            )
        ]

    @staticmethod
    def _unwrap_command_output(output: Any) -> Any:
        """Return the ToolMessage carried by a langgraph ``Command``.

        State-mutating tools (``load_skill``, ``enter_plan_mode``,
        ``exit_plan_mode``) return a ``Command`` whose ToolMessage lives in
        ``update["messages"]`` instead of being returned directly. A
        ``Command`` exposes neither ``content`` nor ``tool_call_id``, so
        without this unwrap the resulting ``ToolResultEvent`` carries
        ``tool_call_id=None`` and the TUI can never correlate it with the
        running ephemeral block — leaving the spinner spinning forever.
        """
        update = getattr(output, "update", None)
        if isinstance(update, dict):
            messages = update.get("messages")
            if isinstance(messages, list):
                for msg in reversed(messages):
                    if isinstance(msg, ToolMessage):
                        return msg
        return output

    def _handle_tool_end(self, event: dict[str, Any]) -> list[AgentStreamEvent]:
        output = event.get("data", {}).get("output")
        if not output:
            return []
        output = self._unwrap_command_output(output)
        tool_content = output.content if hasattr(output, "content") else str(output)
        is_error = (
            isinstance(output, ToolMessage) and getattr(output, "status", "success") == "error"
        )
        out: list[AgentStreamEvent] = []
        if isinstance(output, ToolMessage):
            out.append(MessageEvent(message=output))
        out.append(
            ToolResultEvent(
                content=tool_content,
                tool_call_id=getattr(output, "tool_call_id", None),
                is_error=is_error,
            )
        )
        return out

    def _handle_model_end(self, event: dict[str, Any]) -> list[AgentStreamEvent]:
        output_msg = event.get("data", {}).get("output")
        if output_msg:
            usage = getattr(output_msg, "usage_metadata", None)
            if usage:
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
                cache_read_tokens = 0
                cache_creation_tokens = 0
                # Anthropic direct: cache tokens nested under input_token_details
                details = usage.get("input_token_details", {})
                if isinstance(details, dict):
                    cache_read_tokens += details.get("cache_read", 0)
                    cache_creation_tokens += details.get("cache_creation", 0)
                # Bedrock (langchain_aws): cache tokens at top level after
                # cacheReadInputTokens → cache_read_input_tokens conversion.
                # Bedrock reports input_tokens as the non-cached portion only, so
                # add cache tokens back to input_tokens to match Anthropic's convention
                # (where input_tokens already includes the cached subset).
                bedrock_cache_read = usage.get("cache_read_input_tokens", 0)
                bedrock_cache_write = usage.get("cache_write_input_tokens", 0)
                cache_read_tokens += bedrock_cache_read
                cache_creation_tokens += bedrock_cache_write
                input_tokens += bedrock_cache_read + bedrock_cache_write
                return [
                    UsageEvent(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cache_read_tokens=cache_read_tokens,
                        cache_creation_tokens=cache_creation_tokens,
                    )
                ]
        return []
