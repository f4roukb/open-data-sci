"""Unit tests for opendatasci.streaming.processors (and format_stream_error)."""


import asyncio
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from opendatasci.streaming import (
    AgentTurnStreamProcessor,
    format_stream_error,
)
from opendatasci.streaming.processors import SUBAGENT_TAG

# ---------------------------------------------------------------------------
# format_stream_error
# ---------------------------------------------------------------------------


class TestFormatStreamError:
    def test_connection_error_by_class_name(self) -> None:
        exc = ConnectionError("connection refused")
        result = format_stream_error(exc)
        assert "Connection error" in result
        assert "internet connection" in result

    def test_connection_error_by_keyword_in_message(self) -> None:
        exc = RuntimeError("connection refused to server at port 8080")
        result = format_stream_error(exc)
        assert "Connection error" in result

    def test_connection_error_name_or_service_not_known(self) -> None:
        exc = RuntimeError("name or service not known")
        result = format_stream_error(exc)
        assert "Connection error" in result

    def test_auth_error_by_keyword_api_key(self) -> None:
        exc = RuntimeError("invalid api_key provided")
        result = format_stream_error(exc)
        assert "Authentication error" in result

    def test_auth_error_by_keyword_unauthorized(self) -> None:
        exc = RuntimeError("401 unauthorized")
        result = format_stream_error(exc)
        assert "Authentication error" in result

    def test_auth_error_by_keyword_authentication(self) -> None:
        exc = RuntimeError("authentication failed")
        result = format_stream_error(exc)
        assert "Authentication error" in result

    def test_generic_error_returns_message(self) -> None:
        exc = ValueError("something completely unexpected")
        result = format_stream_error(exc)
        assert "something completely unexpected" in result


# ---------------------------------------------------------------------------
# AgentTurnStreamProcessor
# ---------------------------------------------------------------------------


def _make_chunk(content: object, tool_call_chunks: list | None = None) -> MagicMock:
    chunk = MagicMock()
    chunk.content = content
    chunk.tool_call_chunks = tool_call_chunks or []
    return chunk


def _stream_event(chunk: MagicMock) -> dict:
    return {"event": "on_chat_model_stream", "data": {"chunk": chunk}}


class TestStreamEventProcessor:
    def _proc(self) -> AgentTurnStreamProcessor:
        return AgentTurnStreamProcessor()

    def test_process_unknown_event_returns_empty(self) -> None:
        p = self._proc()
        assert p.process_event({"event": "unknown_event"}) == []

    def test_process_stream_string_content_yields_token(self) -> None:
        p = self._proc()
        chunk = _make_chunk("hello")
        results = p.process_event(_stream_event(chunk))
        assert len(results) == 1
        assert results[0].type == "token"
        assert results[0].content == "hello"

    def test_process_stream_empty_string_yields_nothing(self) -> None:
        p = self._proc()
        chunk = _make_chunk("")
        results = p.process_event(_stream_event(chunk))
        assert results == []

    def test_process_stream_thinking_block(self) -> None:
        p = self._proc()
        chunk = _make_chunk([{"type": "thinking", "thinking": "I think..."}])
        results = p.process_event(_stream_event(chunk))
        assert any(e.type == "reasoning" and "I think" in e.content for e in results)

    def test_process_stream_text_block(self) -> None:
        p = self._proc()
        chunk = _make_chunk([{"type": "text", "text": "my answer"}])
        results = p.process_event(_stream_event(chunk))
        assert any(e.type == "token" and e.content == "my answer" for e in results)

    def test_process_stream_text_block_with_index(self) -> None:
        """Bedrock ConverseStream contentBlockDelta events include an 'index'
        field alongside 'type' and 'text'.  The processor must emit a token
        event for these too, not silently skip them."""
        p = self._proc()
        chunk = _make_chunk([{"type": "text", "text": "bedrock token", "index": 0}])
        results = p.process_event(_stream_event(chunk))
        assert any(e.type == "token" and e.content == "bedrock token" for e in results)

    def test_process_stream_content_block_stop_event_ignored(self) -> None:
        """Bedrock emits contentBlockStop events as chunks with only an 'index'
        key and no 'type'.  These must not produce token events or crash."""
        p = self._proc()
        chunk = _make_chunk([{"index": 0}])
        results = p.process_event(_stream_event(chunk))
        assert not any(e.type == "token" for e in results)

    def test_process_stream_string_block_in_list(self) -> None:
        p = self._proc()
        chunk = _make_chunk(["plain string"])
        results = p.process_event(_stream_event(chunk))
        assert any(e.type == "token" and e.content == "plain string" for e in results)

    def test_process_stream_reasoning_content_block(self) -> None:
        """Bedrock ConverseStream emits reasoning as 'reasoning_content' blocks,
        not 'thinking' blocks.  Both must produce reasoning events."""
        p = self._proc()
        chunk = _make_chunk(
            [
                {
                    "type": "reasoning_content",
                    "reasoning_content": {"type": "text", "text": "bedrock thinking"},
                }
            ]
        )
        results = p.process_event(_stream_event(chunk))
        assert any(e.type == "reasoning" and "bedrock thinking" in e.content for e in results)

    def test_process_stream_reasoning_content_empty_text_ignored(self) -> None:
        p = self._proc()
        chunk = _make_chunk(
            [
                {
                    "type": "reasoning_content",
                    "reasoning_content": {"type": "text", "text": ""},
                }
            ]
        )
        results = p.process_event(_stream_event(chunk))
        assert not any(e.type == "reasoning" for e in results)

    def test_process_stream_unknown_block_type_ignored(self) -> None:
        p = self._proc()
        chunk = _make_chunk([{"type": "redacted_thinking", "data": "x"}])
        results = p.process_event(_stream_event(chunk))
        assert results == []

    def test_process_stream_empty_thinking_block_ignored(self) -> None:
        p = self._proc()
        chunk = _make_chunk([{"type": "thinking", "thinking": ""}])
        results = p.process_event(_stream_event(chunk))
        assert results == []

    def test_process_stream_sets_has_streamed_tokens(self) -> None:
        p = self._proc()
        assert not p._has_streamed_tokens
        p.process_event(_stream_event(_make_chunk("hello")))
        assert p._has_streamed_tokens

    def test_process_tool_end_tool_message(self) -> None:
        p = self._proc()
        tool_msg = ToolMessage(content="result", tool_call_id="tc1")
        event = {"event": "on_tool_end", "data": {"output": tool_msg}}
        results = p.process_event(event)
        tool_result_events = [e for e in results if e.type == "tool_result"]
        assert len(tool_result_events) == 1
        assert tool_result_events[0].content == "result"

    def test_process_tool_end_emits_message_event(self) -> None:
        p = self._proc()
        tool_msg = ToolMessage(content="r", tool_call_id="tc1")
        event = {"event": "on_tool_end", "data": {"output": tool_msg}}
        results = p.process_event(event)
        message_events = [e for e in results if e.type == "message"]
        assert len(message_events) == 1
        assert message_events[0].message is tool_msg

    def test_process_tool_end_empty_output_returns_nothing(self) -> None:
        p = self._proc()
        event = {"event": "on_tool_end", "data": {"output": None}}
        assert p.process_event(event) == []

    def test_process_tool_end_unwraps_command_output(self) -> None:
        """State-mutating tools (load_skill, enter/exit plan mode) return a
        langgraph Command whose ToolMessage is nested in update["messages"].
        The tool_result must carry that ToolMessage's tool_call_id so the TUI
        can correlate it with the running ephemeral block — otherwise the
        spinner never resolves.

        Regression: previously the Command was passed through unchanged, so
        getattr(output, "tool_call_id", None) returned None and the ephemeral
        block spun forever (load_skill / Planning / Done planning)."""
        from langgraph.types import Command

        p = self._proc()
        tool_msg = ToolMessage(content="Plan recorded.", tool_call_id="plan-tc")
        cmd = Command(update={"is_plan_mode": False, "messages": [tool_msg]})
        event = {"event": "on_tool_end", "data": {"output": cmd}}
        results = p.process_event(event)
        tool_results = [e for e in results if e.type == "tool_result"]
        assert len(tool_results) == 1
        assert tool_results[0].content == "Plan recorded."
        assert tool_results[0].tool_call_id == "plan-tc"
        # The unwrapped ToolMessage must also be recorded as generated output.
        from opendatasci.streaming.events import MessageEvent
        assert any(isinstance(e, MessageEvent) and e.message is tool_msg for e in results)

    def test_process_tool_end_command_without_messages_passes_through(self) -> None:
        """A Command carrying no messages list must not crash; the output is
        used as-is (yielding a tool_result with no correlatable id)."""
        from langgraph.types import Command

        p = self._proc()
        cmd = Command(update={"is_plan_mode": True})
        event = {"event": "on_tool_end", "data": {"output": cmd}}
        results = p.process_event(event)
        assert len(results) == 1
        assert results[0].type == "tool_result"

    def test_process_model_end_emits_per_call_tokens(self) -> None:
        p = self._proc()
        msg = MagicMock()
        msg.usage_metadata = {"input_tokens": 10, "output_tokens": 20}
        events = p.process_event({"event": "on_chat_model_end", "data": {"output": msg}})
        assert len(events) == 1
        assert events[0].type == "usage"
        assert events[0].input_tokens == 10
        assert events[0].output_tokens == 20
        assert events[0].cache_read_tokens == 0
        assert events[0].cache_creation_tokens == 0

    def test_process_model_end_emits_per_call_tokens_each_time(self) -> None:
        p = self._proc()
        for _ in range(3):
            msg = MagicMock()
            msg.usage_metadata = {"input_tokens": 5, "output_tokens": 10}
            events = p.process_event({"event": "on_chat_model_end", "data": {"output": msg}})
            assert len(events) == 1
            assert events[0].type == "usage"
            assert events[0].input_tokens == 5
            assert events[0].output_tokens == 10
            assert events[0].cache_read_tokens == 0
            assert events[0].cache_creation_tokens == 0

    def test_process_model_end_emits_cache_tokens(self) -> None:
        p = self._proc()
        msg = MagicMock()
        msg.usage_metadata = {
            "input_tokens": 100,
            "output_tokens": 50,
            "input_token_details": {"cache_read": 40, "cache_creation": 10},
        }
        events = p.process_event({"event": "on_chat_model_end", "data": {"output": msg}})
        assert events[0].cache_read_tokens == 40
        assert events[0].cache_creation_tokens == 10

    def test_process_model_end_cache_tokens_per_call(self) -> None:
        p = self._proc()
        for _ in range(2):
            msg = MagicMock()
            msg.usage_metadata = {
                "input_tokens": 50,
                "output_tokens": 20,
                "input_token_details": {"cache_read": 15},
            }
            events = p.process_event({"event": "on_chat_model_end", "data": {"output": msg}})
            assert events[0].cache_read_tokens == 15

    def test_process_model_end_bedrock_style_cache_tokens(self) -> None:
        """Bedrock (langchain_aws) puts cache counts at the top level of
        usage_metadata as cache_read_input_tokens / cache_write_input_tokens
        after camelCase→snake_case conversion of the Converse API response."""
        p = self._proc()
        msg = MagicMock()
        msg.usage_metadata = {
            "input_tokens": 600,
            "output_tokens": 141,
            "total_tokens": 7371,
            "cache_read_input_tokens": 6630,
            "cache_read_input_token_count": 6630,
            "cache_write_input_tokens": 0,
            "cache_write_input_token_count": 0,
        }
        events = p.process_event({"event": "on_chat_model_end", "data": {"output": msg}})
        assert events[0].cache_read_tokens == 6630
        assert events[0].cache_creation_tokens == 0
        # input_tokens must be normalized to include the cached portion so the
        # cache percentage denominator (input + output) reflects the full context.
        assert events[0].input_tokens == 7230

    def test_process_model_end_bedrock_cache_tokens_per_call(self) -> None:
        p = self._proc()
        for _ in range(2):
            msg = MagicMock()
            msg.usage_metadata = {
                "input_tokens": 300,
                "output_tokens": 70,
                "cache_read_input_tokens": 3000,
                "cache_write_input_tokens": 500,
            }
            events = p.process_event({"event": "on_chat_model_end", "data": {"output": msg}})
            # Each call: 300 + 3000 (cache_read) + 500 (cache_write) = 3800 normalized input
            assert events[0].input_tokens == 3800
            assert events[0].cache_read_tokens == 3000
            assert events[0].cache_creation_tokens == 500

    def test_process_model_end_non_dict_input_token_details_ignored(self) -> None:
        p = self._proc()
        msg = MagicMock()
        msg.usage_metadata = {
            "input_tokens": 10,
            "output_tokens": 5,
            "input_token_details": [1, 2, 3],
        }
        events = p.process_event({"event": "on_chat_model_end", "data": {"output": msg}})
        assert events[0].cache_read_tokens == 0

    def test_process_model_end_no_usage_metadata(self) -> None:
        p = self._proc()
        msg = MagicMock()
        msg.usage_metadata = None
        events = p.process_event({"event": "on_chat_model_end", "data": {"output": msg}})
        assert events == []

    def test_process_chain_end_falls_back_to_msg_content_when_no_tokens_streamed_string(
        self,
    ) -> None:
        """When no tokens were streamed the fallback emits a token event from msg.content."""
        p = self._proc()
        ai_msg = AIMessage(content="Final answer from model")
        event = {
            "event": "on_chain_end",
            "name": "agent",
            "data": {"output": {"messages": [ai_msg]}},
        }
        results = p.process_event(event)
        token_events = [e for e in results if e.type == "token"]
        assert len(token_events) == 1
        assert token_events[0].content == "Final answer from model"

    def test_process_chain_end_falls_back_to_msg_content_when_no_tokens_streamed_list_blocks(
        self,
    ) -> None:
        """Fallback also works when msg.content is a list of typed blocks."""
        p = self._proc()
        ai_msg = AIMessage(content=[{"type": "text", "text": "Answer in blocks"}])
        event = {
            "event": "on_chain_end",
            "name": "agent",
            "data": {"output": {"messages": [ai_msg]}},
        }
        results = p.process_event(event)
        token_events = [e for e in results if e.type == "token"]
        assert len(token_events) == 1
        assert token_events[0].content == "Answer in blocks"

    def test_process_chain_end_fallback_ignores_non_text_blocks(self) -> None:
        """Non-text blocks (e.g. thinking) must not appear in the fallback token."""
        p = self._proc()
        ai_msg = AIMessage(
            content=[
                {"type": "thinking", "thinking": "private reasoning"},
                {"type": "text", "text": "Visible answer"},
            ]
        )
        event = {
            "event": "on_chain_end",
            "name": "agent",
            "data": {"output": {"messages": [ai_msg]}},
        }
        results = p.process_event(event)
        token_events = [e for e in results if e.type == "token"]
        assert len(token_events) == 1
        assert token_events[0].content == "Visible answer"
        assert "private reasoning" not in token_events[0].content

    def test_process_chain_end_fallback_no_token_event_when_content_empty(self) -> None:
        """No token event should be emitted when the fallback text is empty."""
        p = self._proc()
        ai_msg = AIMessage(content="")
        event = {
            "event": "on_chain_end",
            "name": "agent",
            "data": {"output": {"messages": [ai_msg]}},
        }
        results = p.process_event(event)
        assert not any(e.type == "token" for e in results)

    def test_process_chain_end_no_fallback_when_tokens_already_streamed(self) -> None:
        """When tokens were already streamed no extra token event is emitted."""
        p = self._proc()
        p._has_streamed_tokens = True
        ai_msg = AIMessage(content="different content on msg")
        event = {
            "event": "on_chain_end",
            "name": "agent",
            "data": {"output": {"messages": [ai_msg]}},
        }
        results = p.process_event(event)
        assert not any(e.type == "token" for e in results)

    def test_process_chain_end_resets_has_streamed_tokens(self) -> None:
        p = self._proc()
        p._has_streamed_tokens = True
        ai_msg = AIMessage(content="text")
        event = {
            "event": "on_chain_end",
            "name": "agent",
            "data": {"output": {"messages": [ai_msg]}},
        }
        p.process_event(event)
        assert not p._has_streamed_tokens

    def test_process_chain_end_adds_ai_message_to_generated(self) -> None:
        p = self._proc()
        ai_msg = AIMessage(content="hi")
        event = {
            "event": "on_chain_end",
            "name": "agent",
            "data": {"output": {"messages": [ai_msg]}},
        }
        results = p.process_event(event)
        from opendatasci.streaming.events import MessageEvent
        assert any(isinstance(e, MessageEvent) and e.message is ai_msg for e in results)

    def test_chain_end_with_non_ai_message_ignored(self) -> None:
        p = self._proc()
        human_msg = HumanMessage(content="hi")
        event = {
            "event": "on_chain_end",
            "name": "agent",
            "data": {"output": {"messages": [human_msg]}},
        }
        results = p.process_event(event)
        assert not any(e.type == "message" for e in results)

    def test_chain_end_wrong_name_ignored(self) -> None:
        p = self._proc()
        p._has_streamed_tokens = True
        ai_msg = AIMessage(content="text")
        event = {
            "event": "on_chain_end",
            "name": "tools",  # Not "agent"
            "data": {"output": {"messages": [ai_msg]}},
        }
        results = p.process_event(event)
        assert not any(e.type == "message" for e in results)

    def test_subagent_tagged_stream_event_dropped(self) -> None:
        """Worker chat-model stream chunks must not produce token events."""
        p = self._proc()
        chunk = _make_chunk("worker text leak")
        event = _stream_event(chunk)
        event["tags"] = [SUBAGENT_TAG]
        assert p.process_event(event) == []
        assert not p._has_streamed_tokens

    def test_subagent_tagged_thinking_block_dropped(self) -> None:
        p = self._proc()
        chunk = _make_chunk([{"type": "thinking", "thinking": "worker reasoning"}])
        event = _stream_event(chunk)
        event["tags"] = [SUBAGENT_TAG]
        assert p.process_event(event) == []

    def test_subagent_tagged_tool_end_dropped(self) -> None:
        """Worker-internal tool calls must not bubble up as tool_result events."""
        p = self._proc()
        tool_msg = ToolMessage(content="worker tool output", tool_call_id="worker-tc")
        event = {
            "event": "on_tool_end",
            "data": {"output": tool_msg},
            "tags": [SUBAGENT_TAG],
        }
        results = p.process_event(event)
        assert results == []

    def test_subagent_tagged_chain_end_not_processed(self) -> None:
        p = self._proc()
        ai_msg = AIMessage(content="worker final answer")
        event = {
            "event": "on_chain_end",
            "name": "agent",
            "data": {"output": {"messages": [ai_msg]}},
            "tags": [SUBAGENT_TAG],
        }
        assert p.process_event(event) == []

    def test_subagent_tag_among_other_tags_still_filtered(self) -> None:
        p = self._proc()
        chunk = _make_chunk("leak")
        event = _stream_event(chunk)
        event["tags"] = ["other-tag", SUBAGENT_TAG, "another"]
        assert p.process_event(event) == []

    def test_main_agent_event_without_tags_still_processed(self) -> None:
        """Sanity check — events without the subagent tag flow through normally."""
        p = self._proc()
        chunk = _make_chunk("hello")
        event = _stream_event(chunk)
        event["tags"] = ["unrelated-tag"]
        results = p.process_event(event)
        assert any(e.type == "token" and e.content == "hello" for e in results)

    def test_tool_communication_emitted_from_chain_end_when_not_streamed(self) -> None:
        """tool_communication must be emitted from the final AIMessage args even
        when the streaming path produced no tool_call_chunks (e.g. the LLM
        doesn't stream incrementally or the communication value only becomes
        available at on_chain_end).

        Regression: previously _handle_tool_call never emitted tool_communication,
        so the communication was silently dropped whenever _handle_stream didn't
        capture it first."""
        p = self._proc()
        ai_msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "web_search",
                    "args": {"communication": "Searching for data", "query": "X"},
                    "id": "tc1",
                }
            ],
        )
        results = p.process_event(
            {
                "event": "on_chain_end",
                "name": "agent",
                "data": {"output": {"messages": [ai_msg]}},
            }
        )

        comm_events = [e for e in results if e.type == "tool_communication"]
        call_events = [e for e in results if e.type == "tool_call"]

        assert len(comm_events) == 1, "Expected one tool_communication event from chain_end"
        assert comm_events[0].content == "Searching for data"
        assert comm_events[0].tool_call_id == "tc1"
        assert comm_events[0].tool_name == "web_search"

        assert len(call_events) == 1
        # Communication must precede the tool_call so the controller can
        # pre-mount the ephemeral block before upgrading it.
        assert results.index(comm_events[0]) < results.index(call_events[0])

    def test_tool_communication_not_duplicated_when_streaming_already_emitted_it(self) -> None:
        """If _handle_stream already emitted tool_communication during streaming,
        _handle_chain_end must not emit a duplicate."""
        p = self._proc()

        # Streaming captures the communication from incremental chunks.
        chunk = _make_chunk(
            "",
            [
                {
                    "name": "web_search",
                    "id": "tc1",
                    "args": '{"communication": "Searching", "query": "X"}',
                    "index": 0,
                }
            ],
        )
        streaming_events = p.process_event(_stream_event(chunk))
        assert any(e.type == "tool_communication" for e in streaming_events)

        # chain_end fires with the same complete args.
        ai_msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "web_search",
                    "args": {"communication": "Searching", "query": "X"},
                    "id": "tc1",
                }
            ],
        )
        chain_end_results = p.process_event(
            {
                "event": "on_chain_end",
                "name": "agent",
                "data": {"output": {"messages": [ai_msg]}},
            }
        )

        # No duplicate — the streaming value was already recorded in _last_communication.
        assert not any(e.type == "tool_communication" for e in chain_end_results)

    def test_tool_communication_emitted_for_single_tool_call(self) -> None:
        """A single tool call with a communication arg emits one tool_communication event."""
        p = self._proc()
        chunk1 = _make_chunk("", [{"name": "web_search", "id": "tc1", "args": "", "index": 0}])
        p.process_event(_stream_event(chunk1))

        chunk2 = _make_chunk("", [{"args": '{"communication": "Searching now"}', "index": 0}])
        events = p.process_event(_stream_event(chunk2))

        comm_events = [e for e in events if e.type == "tool_communication"]
        assert len(comm_events) == 1
        assert comm_events[0].content == "Searching now"
        assert comm_events[0].tool_call_id == "tc1"
        assert comm_events[0].tool_name == "web_search"

    def test_tool_communication_emitted_for_each_parallel_tool_call(self) -> None:
        """Parallel tool calls whose arg chunks carry an index field must each
        receive their own tool_communication event with the correct content.

        Regression test: previously the code always inspected
        ``_pending_tool_calls[-1]``, so when a second tool call started before
        the first one's args arrived the first call's communication was silently
        dropped and the second call received the wrong content."""
        p = self._proc()

        # Both tool calls start (first chunks carry name + id + index).
        chunk_tc1_start = _make_chunk(
            "", [{"name": "web_search", "id": "tc1", "args": "", "index": 0}]
        )
        chunk_tc2_start = _make_chunk(
            "", [{"name": "fetch_url", "id": "tc2", "args": "", "index": 1}]
        )
        p.process_event(_stream_event(chunk_tc1_start))
        p.process_event(_stream_event(chunk_tc2_start))

        # tc1 args arrive (index=0) while tc2 is already in pending list.
        chunk_tc1_args = _make_chunk(
            "", [{"args": '{"communication": "Searching for data"}', "index": 0}]
        )
        events_tc1 = p.process_event(_stream_event(chunk_tc1_args))

        # tc2 args arrive (index=1).
        chunk_tc2_args = _make_chunk(
            "", [{"args": '{"communication": "Fetching the page"}', "index": 1}]
        )
        events_tc2 = p.process_event(_stream_event(chunk_tc2_args))

        all_comm = [e for e in events_tc1 + events_tc2 if e.type == "tool_communication"]
        assert len(all_comm) == 2, f"Expected 2 tool_communication events, got {len(all_comm)}"

        by_id = {e.tool_call_id: e for e in all_comm}
        assert "tc1" in by_id, "No communication event emitted for tc1"
        assert "tc2" in by_id, "No communication event emitted for tc2"
        assert by_id["tc1"].content == "Searching for data"
        assert by_id["tc2"].content == "Fetching the page"
        assert by_id["tc1"].tool_name == "web_search"
        assert by_id["tc2"].tool_name == "fetch_url"

    def test_tool_communication_without_index_falls_back_to_last(self) -> None:
        """When tool_call_chunks carry no index (Anthropic-style sequential streaming),
        the existing behaviour of routing to the last pending tool call is preserved."""
        p = self._proc()

        # Tool call starts without an index field.
        chunk_start = _make_chunk("", [{"name": "web_search", "id": "tc1", "args": ""}])
        p.process_event(_stream_event(chunk_start))

        # Arg chunk also has no index — should still update the (only) pending call.
        chunk_args = _make_chunk("", [{"args": '{"communication": "Searching"}'}])
        events = p.process_event(_stream_event(chunk_args))

        comm_events = [e for e in events if e.type == "tool_communication"]
        assert len(comm_events) == 1
        assert comm_events[0].content == "Searching"
        assert comm_events[0].tool_call_id == "tc1"
        assert comm_events[0].tool_name == "web_search"

    def test_tool_communication_deduped_within_same_tool_call(self) -> None:
        """The same communication value must not be emitted twice for the same tool call."""
        p = self._proc()

        chunk_start = _make_chunk("", [{"name": "web_search", "id": "tc1", "args": "", "index": 0}])
        p.process_event(_stream_event(chunk_start))

        # Two chunks that each contain the complete communication value.
        chunk_a = _make_chunk("", [{"args": '{"communication": "Searching"}', "index": 0}])
        chunk_b = _make_chunk(
            "", [{"args": '{"communication": "Searching", "query": "x"}', "index": 0}]
        )
        events_a = p.process_event(_stream_event(chunk_a))
        events_b = p.process_event(_stream_event(chunk_b))

        comm_events = [e for e in events_a + events_b if e.type == "tool_communication"]
        assert len(comm_events) == 1, "Communication value must be emitted only once"

    def test_stream_emits_incremental_usage_when_input_tokens_in_chunk(self) -> None:
        """When a chunk carries usage_metadata with input_tokens, text token chunks
        also emit a usage event with an estimated output token count."""
        p = self._proc()
        chunk = _make_chunk("hello world")
        chunk.usage_metadata = {"input_tokens": 1000}
        results = p.process_event(_stream_event(chunk))
        usage_events = [e for e in results if e.type == "usage"]
        assert len(usage_events) == 1
        assert usage_events[0].input_tokens == 1000
        # "hello world" = 11 chars → max(1, 11 // 4) = 2
        assert usage_events[0].output_tokens == max(1, len("hello world") // 4)

    def test_stream_no_incremental_usage_without_input_tokens(self) -> None:
        """Without usage_metadata providing input_tokens, no usage event is emitted
        during streaming — only the final event from _handle_model_end fires."""
        p = self._proc()
        chunk = _make_chunk("hello")
        results = p.process_event(_stream_event(chunk))
        assert not any(e.type == "usage" for e in results)

    def test_stream_incremental_usage_output_chars_accumulate(self) -> None:
        """Estimated output token count grows across consecutive text chunks."""
        p = self._proc()
        chunk1 = _make_chunk("hello")
        chunk1.usage_metadata = {"input_tokens": 5000}
        results1 = p.process_event(_stream_event(chunk1))

        chunk2 = _make_chunk(" world")
        chunk2.usage_metadata = {}  # no input_tokens in subsequent chunks
        results2 = p.process_event(_stream_event(chunk2))

        usage1 = next(e for e in results1 if e.type == "usage")
        usage2 = next(e for e in results2 if e.type == "usage")
        assert usage2.output_tokens >= usage1.output_tokens

    def test_stream_incremental_usage_not_emitted_for_non_text_chunks(self) -> None:
        """Chunks with no text tokens (e.g. pure tool-call args) do not emit usage."""
        p = self._proc()
        # Seed input tokens with a text chunk first
        seed = _make_chunk("hi")
        seed.usage_metadata = {"input_tokens": 100}
        p.process_event(_stream_event(seed))

        # Now send a chunk with only tool_call_chunks and no text
        chunk = _make_chunk("")
        chunk.usage_metadata = {}
        chunk.tool_call_chunks = [{"name": "web_search", "id": "tc1", "args": "", "index": 0}]
        results = p.process_event(_stream_event(chunk))
        assert not any(e.type == "usage" for e in results)

    def test_stream_incremental_usage_resets_on_chain_end(self) -> None:
        """_stream_input_tokens and _stream_output_chars reset after chain_end
        so the next model invocation starts fresh."""
        p = self._proc()
        chunk = _make_chunk("some text")
        chunk.usage_metadata = {"input_tokens": 100}
        p.process_event(_stream_event(chunk))
        assert p._stream_input_tokens == 100
        assert p._stream_output_chars > 0

        ai_msg = AIMessage(content="some text")
        p.process_event(
            {
                "event": "on_chain_end",
                "name": "agent",
                "data": {"output": {"messages": [ai_msg]}},
            }
        )
        assert p._stream_input_tokens is None
        assert p._stream_output_chars == 0

    def test_stream_incremental_usage_input_tokens_captured_once(self) -> None:
        """input_tokens is only captured from the first chunk that provides it;
        subsequent chunks with usage_metadata do not overwrite it."""
        p = self._proc()
        chunk1 = _make_chunk("first")
        chunk1.usage_metadata = {"input_tokens": 200}
        p.process_event(_stream_event(chunk1))

        chunk2 = _make_chunk(" second")
        chunk2.usage_metadata = {"input_tokens": 999}  # should be ignored
        results2 = p.process_event(_stream_event(chunk2))

        usage2 = next(e for e in results2 if e.type == "usage")
        assert usage2.input_tokens == 200


# ---------------------------------------------------------------------------
# AgentTurnStreamProcessor — worker_event (on_custom_event) handling
# ---------------------------------------------------------------------------


class TestWorkerEventHandling:
    def _proc(self) -> AgentTurnStreamProcessor:
        return AgentTurnStreamProcessor()

    def _worker_event(self, data: dict) -> dict:
        return {"event": "on_custom_event", "name": "worker_event", "data": data}

    def test_worker_done_event_type(self) -> None:
        events = self._proc().process_event(
            self._worker_event({"event_type": "worker_done", "worker_idx": 0, "success": True})
        )
        assert len(events) == 1
        assert events[0].type == "worker_done"

    def test_worker_done_carries_idx_and_success(self) -> None:
        events = self._proc().process_event(
            self._worker_event({"event_type": "worker_done", "worker_idx": 2, "success": False})
        )
        assert events[0].worker_idx == 2
        assert events[0].success is False

    def test_worker_done_defaults_success_to_true(self) -> None:
        events = self._proc().process_event(
            self._worker_event({"event_type": "worker_done", "worker_idx": 1})
        )
        assert events[0].success is True

    def test_lifecycle_event_becomes_subagent_event(self) -> None:
        events = self._proc().process_event(
            self._worker_event({"event_type": "worker_started", "worker_idx": 1, "content": "task"})
        )
        assert len(events) == 1
        assert events[0].type == "subagent_event"
        assert events[0].event_type == "worker_started"
        assert events[0].content == "task"

    def test_subagent_event_carries_worker_idx(self) -> None:
        events = self._proc().process_event(
            self._worker_event({"event_type": "worker_tool_call", "worker_idx": 3, "content": ""})
        )
        assert events[0].worker_idx == 3

    def test_unknown_name_not_handled(self) -> None:
        events = self._proc().process_event(
            {"event": "on_custom_event", "name": "other_event", "data": {}}
        )
        assert events == []

    def test_worker_event_not_filtered_by_subagent_tag(self) -> None:
        """on_custom_event dispatched with the outer config carries no subagent tag."""
        events = self._proc().process_event(
            {
                "event": "on_custom_event",
                "name": "worker_event",
                "tags": [],
                "data": {"event_type": "worker_done", "worker_idx": 0, "success": True},
            }
        )
        assert any(e.type == "worker_done" for e in events)


# ---------------------------------------------------------------------------
# AgentTurnStreamProcessor — spawn_workers tool_call handling
# ---------------------------------------------------------------------------


def _chain_end_with_tool_calls(tool_calls: list[dict]) -> dict:
    """Build an on_chain_end event with an AIMessage carrying the given tool_calls."""
    msg = AIMessage(content="", tool_calls=tool_calls)  # type: ignore[arg-type]
    return {"event": "on_chain_end", "name": "agent", "data": {"output": {"messages": [msg]}}}


class TestSpawnWorkersToolCall:
    def _proc(self) -> AgentTurnStreamProcessor:
        return AgentTurnStreamProcessor()

    def test_spawn_workers_emits_tool_call_with_spawn_workers_metadata(self) -> None:
        from opendatasci.tools import ToolName

        tc = {
            "name": ToolName.SPAWN_WORKERS,
            "args": {"subtasks": [{"summary": "Analyse"}, {"summary": "Plot"}]},
            "id": "sw1",
        }
        results = self._proc().process_event(_chain_end_with_tool_calls([tc]))
        tool_calls = [e for e in results if e.type == "tool_call"]
        assert len(tool_calls) == 1
        assert tool_calls[0].tool == ToolName.SPAWN_WORKERS

    def test_spawn_workers_tool_metadata_survives_str_roundtrip(self) -> None:
        """Regression: ``ToolName`` is a (str, Enum), so ``str(member)`` yields
        ``'ToolName.SPAWN_WORKERS'`` rather than ``'spawn_workers'``. The
        presenter does ``str(event.tool)`` before comparing against
        ``ToolName.SPAWN_WORKERS`` — storing the enum here would cause that
        comparison to silently fail and the worker rows would never render."""
        from opendatasci.tools import ToolName

        tc = {
            "name": ToolName.SPAWN_WORKERS,
            "args": {"subtasks": [{"summary": "Analyse"}]},
            "id": "sw1",
        }
        results = self._proc().process_event(_chain_end_with_tool_calls([tc]))
        tool_call = next(e for e in results if e.type == "tool_call")
        # This is the exact comparison the presenter performs.
        assert str(tool_call.tool) == ToolName.SPAWN_WORKERS

    def test_spawn_workers_extracts_summaries_from_subtasks(self) -> None:
        from opendatasci.tools import ToolName

        tc = {
            "name": ToolName.SPAWN_WORKERS,
            "args": {"subtasks": [{"summary": "Train"}, {"summary": "Evaluate"}]},
            "id": "sw1",
        }
        results = self._proc().process_event(_chain_end_with_tool_calls([tc]))
        tool_call = next(e for e in results if e.type == "tool_call")
        assert tool_call.worker_summaries == ["Train", "Evaluate"]

    def test_spawn_workers_fills_default_summary_when_missing(self) -> None:
        """A subtask dict without ``summary`` must fall back to ``ParallelWorkerAgent {i+1}``."""
        from opendatasci.tools import ToolName

        tc = {
            "name": ToolName.SPAWN_WORKERS,
            "args": {"subtasks": [{"summary": "Train"}, {}]},  # second subtask has no summary
            "id": "sw1",
        }
        results = self._proc().process_event(_chain_end_with_tool_calls([tc]))
        tool_call = next(e for e in results if e.type == "tool_call")
        assert tool_call.worker_summaries == ["Train", "ParallelWorkerAgent 2"]

    def test_spawn_workers_non_dict_subtask_falls_back_to_default(self) -> None:
        """If ``subtasks`` contains non-dict entries the placeholder name is used."""
        from opendatasci.tools import ToolName

        tc = {
            "name": ToolName.SPAWN_WORKERS,
            "args": {"subtasks": ["not a dict"]},
            "id": "sw1",
        }
        results = self._proc().process_event(_chain_end_with_tool_calls([tc]))
        tool_call = next(e for e in results if e.type == "tool_call")
        assert tool_call.worker_summaries == ["ParallelWorkerAgent 1"]

    def test_spawn_workers_empty_subtasks_yields_empty_summaries(self) -> None:
        from opendatasci.tools import ToolName

        tc = {
            "name": ToolName.SPAWN_WORKERS,
            "args": {"subtasks": []},
            "id": "sw1",
        }
        results = self._proc().process_event(_chain_end_with_tool_calls([tc]))
        tool_call = next(e for e in results if e.type == "tool_call")
        assert tool_call.worker_summaries == []

    def test_spawn_workers_metadata_carries_tool_call_id(self) -> None:
        from opendatasci.tools import ToolName

        tc = {
            "name": ToolName.SPAWN_WORKERS,
            "args": {"subtasks": []},
            "id": "sw-abc",
        }
        results = self._proc().process_event(_chain_end_with_tool_calls([tc]))
        tool_call = next(e for e in results if e.type == "tool_call")
        assert tool_call.tool_call_id == "sw-abc"

    def test_spawn_workers_no_summary_key_in_metadata(self) -> None:
        """The spawn_workers branch returns early before the generic ``summary``
        is computed — only ``worker_summaries`` should be present."""
        from opendatasci.tools import ToolName

        tc = {
            "name": ToolName.SPAWN_WORKERS,
            "args": {"subtasks": [{"summary": "A"}]},
            "id": "sw1",
        }
        results = self._proc().process_event(_chain_end_with_tool_calls([tc]))
        tool_call = next(e for e in results if e.type == "tool_call")
        # spawn_workers uses worker_summaries, not summary
        assert tool_call.summary == "" and tool_call.worker_summaries == ["A"]


# ---------------------------------------------------------------------------
# AgentTurnStreamProcessor — edge cases in _handle_stream
# ---------------------------------------------------------------------------


class TestHandleStreamEdgeCases:
    def _proc(self) -> AgentTurnStreamProcessor:
        return AgentTurnStreamProcessor()

    def test_stream_event_without_chunk_returns_empty(self) -> None:
        """When the stream event carries no ``chunk`` (e.g. metadata-only events
        from some providers), no StreamEvents must be produced."""
        results = self._proc().process_event({"event": "on_chat_model_stream", "data": {}})
        assert results == []

    def test_stream_block_with_unexpected_python_type_is_logged_and_ignored(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Blocks that are neither dict nor string fall through to a DEBUG log
        line and produce no StreamEvents — defensive against future provider quirks."""
        import logging

        proc = self._proc()
        # 42 is an int — not a dict, not a string.
        chunk = _make_chunk([42])
        with caplog.at_level(logging.DEBUG, logger="opendatasci.streaming.processors"):
            results = proc.process_event(_stream_event(chunk))
        assert results == []
        assert any("Unhandled stream block type" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# AgentTurnStreamProcessor — chain_end fallback handling of string blocks
# ---------------------------------------------------------------------------


class TestChainEndStringBlockFallback:
    def test_string_block_appended_to_fallback_text(self) -> None:
        """Some providers emit content lists containing bare strings in addition
        to typed dict blocks.  The fallback path must include those strings so
        the response is not silently dropped."""
        p = AgentTurnStreamProcessor()
        ai_msg = AIMessage(
            content=[
                "leading plain text. ",
                {"type": "text", "text": "typed block."},
                "trailing plain text.",
            ]
        )
        event = {
            "event": "on_chain_end",
            "name": "agent",
            "data": {"output": {"messages": [ai_msg]}},
        }
        results = p.process_event(event)
        token_events = [e for e in results if e.type == "token"]
        assert len(token_events) == 1
        assert "leading plain text" in token_events[0].content
        assert "trailing plain text" in token_events[0].content


# ---------------------------------------------------------------------------
# _extract_content_events
# ---------------------------------------------------------------------------


class TestExtractContentEvents:
    def _proc(self) -> AgentTurnStreamProcessor:
        return AgentTurnStreamProcessor()

    def test_string_content_yields_token(self) -> None:
        p = self._proc()
        events = p._extract_content_events("hello")
        assert len(events) == 1
        assert events[0].type == "token"
        assert events[0].content == "hello"

    def test_string_content_sets_has_streamed_tokens(self) -> None:
        p = self._proc()
        p._extract_content_events("hi")
        assert p._has_streamed_tokens

    def test_empty_string_yields_nothing(self) -> None:
        p = self._proc()
        assert p._extract_content_events("") == []
        assert not p._has_streamed_tokens

    def test_text_block_yields_token(self) -> None:
        p = self._proc()
        events = p._extract_content_events([{"type": "text", "text": "answer"}])
        assert len(events) == 1
        assert events[0].type == "token"
        assert events[0].content == "answer"

    def test_text_block_sets_has_streamed_tokens(self) -> None:
        p = self._proc()
        p._extract_content_events([{"type": "text", "text": "x"}])
        assert p._has_streamed_tokens

    def test_thinking_block_yields_reasoning(self) -> None:
        p = self._proc()
        events = p._extract_content_events([{"type": "thinking", "thinking": "hmm"}])
        assert len(events) == 1
        assert events[0].type == "reasoning"
        assert events[0].content == "hmm"

    def test_thinking_block_does_not_set_has_streamed_tokens(self) -> None:
        p = self._proc()
        p._extract_content_events([{"type": "thinking", "thinking": "hmm"}])
        assert not p._has_streamed_tokens

    def test_reasoning_content_block_yields_reasoning(self) -> None:
        p = self._proc()
        events = p._extract_content_events(
            [{"type": "reasoning_content", "reasoning_content": {"text": "bedrock think"}}]
        )
        assert len(events) == 1
        assert events[0].type == "reasoning"
        assert events[0].content == "bedrock think"

    def test_empty_thinking_block_yields_nothing(self) -> None:
        p = self._proc()
        assert p._extract_content_events([{"type": "thinking", "thinking": ""}]) == []

    def test_string_in_list_yields_token(self) -> None:
        p = self._proc()
        events = p._extract_content_events(["plain"])
        assert len(events) == 1
        assert events[0].type == "token"
        assert events[0].content == "plain"

    def test_unknown_block_type_ignored(self) -> None:
        p = self._proc()
        assert p._extract_content_events([{"type": "redacted_thinking", "data": "x"}]) == []

    def test_non_string_non_dict_block_logged_and_ignored(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        p = self._proc()
        with caplog.at_level(logging.DEBUG, logger="opendatasci.streaming.processors"):
            events = p._extract_content_events([42])
        assert events == []
        assert any("Unhandled stream block type" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# _resolve_tool_call_target
# ---------------------------------------------------------------------------


class TestResolveToolCallTarget:
    def _proc(self) -> AgentTurnStreamProcessor:
        return AgentTurnStreamProcessor()

    def test_returns_none_when_no_pending_calls(self) -> None:
        p = self._proc()
        assert p._resolve_tool_call_target(0) is None
        assert p._resolve_tool_call_target(None) is None

    def test_returns_last_when_index_is_none(self) -> None:
        p = self._proc()
        tc1: dict = {"name": "a", "args": "", "id": "1", "index": 0}
        tc2: dict = {"name": "b", "args": "", "id": "2", "index": 1}
        p._pending_tool_calls = [tc1, tc2]
        assert p._resolve_tool_call_target(None) is tc2

    def test_returns_matching_by_index(self) -> None:
        p = self._proc()
        tc1: dict = {"name": "a", "args": "", "id": "1", "index": 0}
        tc2: dict = {"name": "b", "args": "", "id": "2", "index": 1}
        p._pending_tool_calls = [tc1, tc2]
        assert p._resolve_tool_call_target(0) is tc1
        assert p._resolve_tool_call_target(1) is tc2

    def test_falls_back_to_last_when_index_not_found(self) -> None:
        p = self._proc()
        tc: dict = {"name": "a", "args": "", "id": "1", "index": 0}
        p._pending_tool_calls = [tc]
        assert p._resolve_tool_call_target(99) is tc


# ---------------------------------------------------------------------------
# _accumulate_tool_call_chunks
# ---------------------------------------------------------------------------


class TestAccumulateToolCallChunks:
    def _proc(self) -> AgentTurnStreamProcessor:
        return AgentTurnStreamProcessor()

    def test_new_named_chunk_appended_to_pending(self) -> None:
        p = self._proc()
        p._accumulate_tool_call_chunks(
            [{"name": "web_search", "id": "tc1", "args": "", "index": 0}], []
        )
        assert len(p._pending_tool_calls) == 1
        assert p._pending_tool_calls[0]["name"] == "web_search"

    def test_arg_chunk_appended_to_target_by_index(self) -> None:
        p = self._proc()
        p._pending_tool_calls = [{"name": "a", "args": '{"x":', "id": "1", "index": 0}]
        p._accumulate_tool_call_chunks([{"args": '"val"}', "index": 0}], [])
        assert p._pending_tool_calls[0]["args"] == '{"x":"val"}'

    def test_arg_chunk_falls_back_to_last_when_no_index(self) -> None:
        p = self._proc()
        p._pending_tool_calls = [{"name": "a", "args": "foo", "id": "1", "index": None}]
        p._accumulate_tool_call_chunks([{"args": "bar"}], [])
        assert p._pending_tool_calls[0]["args"] == "foobar"

    def test_communication_event_emitted(self) -> None:
        p = self._proc()
        out: list = []
        p._accumulate_tool_call_chunks(
            [{"name": "web_search", "id": "tc1", "args": "", "index": 0}], out
        )
        p._accumulate_tool_call_chunks(
            [{"args": '{"communication": "Searching"}', "index": 0}], out
        )
        comm = [e for e in out if e.type == "tool_communication"]
        assert len(comm) == 1
        assert comm[0].content == "Searching"
        assert comm[0].tool_call_id == "tc1"

    def test_communication_deduped_for_same_tool_call(self) -> None:
        p = self._proc()
        out: list = []
        p._accumulate_tool_call_chunks(
            [{"name": "web_search", "id": "tc1", "args": "", "index": 0}], out
        )
        p._accumulate_tool_call_chunks(
            [{"args": '{"communication": "Searching"}', "index": 0}], out
        )
        p._accumulate_tool_call_chunks(
            [{"args": '{"communication": "Searching", "q": "x"}', "index": 0}], out
        )
        comm = [e for e in out if e.type == "tool_communication"]
        assert len(comm) == 1

    def test_communication_emitted_independently_for_parallel_calls(self) -> None:
        p = self._proc()
        out: list = []
        p._accumulate_tool_call_chunks(
            [{"name": "a", "id": "t1", "args": "", "index": 0}], out
        )
        p._accumulate_tool_call_chunks(
            [{"name": "b", "id": "t2", "args": "", "index": 1}], out
        )
        p._accumulate_tool_call_chunks(
            [{"args": '{"communication": "Alpha"}', "index": 0}], out
        )
        p._accumulate_tool_call_chunks(
            [{"args": '{"communication": "Beta"}', "index": 1}], out
        )
        comm = {e.tool_call_id: e.content for e in out if e.type == "tool_communication"}
        assert comm == {"t1": "Alpha", "t2": "Beta"}


# ---------------------------------------------------------------------------
# _update_stream_usage
# ---------------------------------------------------------------------------


class TestUpdateStreamUsage:
    def _proc(self) -> AgentTurnStreamProcessor:
        return AgentTurnStreamProcessor()

    def _chunk_with_usage(self, input_tokens: int | None = None) -> MagicMock:
        chunk = MagicMock()
        chunk.usage_metadata = {"input_tokens": input_tokens} if input_tokens is not None else {}
        return chunk

    def test_captures_input_tokens_from_first_chunk(self) -> None:
        p = self._proc()
        p._update_stream_usage(self._chunk_with_usage(500), [])
        assert p._stream_input_tokens == 500

    def test_does_not_overwrite_stream_input_tokens(self) -> None:
        p = self._proc()
        p._update_stream_usage(self._chunk_with_usage(500), [])
        p._update_stream_usage(self._chunk_with_usage(999), [])
        assert p._stream_input_tokens == 500

    def test_emits_usage_event_when_token_chars_present(self) -> None:
        from opendatasci.streaming.events import TokenEvent, UsageEvent

        p = self._proc()
        p._stream_input_tokens = 100
        out = [TokenEvent(content="hello")]
        p._update_stream_usage(self._chunk_with_usage(), out)
        usage = [e for e in out if isinstance(e, UsageEvent)]
        assert len(usage) == 1
        assert usage[0].input_tokens == 100

    def test_does_not_emit_usage_without_token_chars(self) -> None:
        p = self._proc()
        p._stream_input_tokens = 100
        out: list = []
        p._update_stream_usage(self._chunk_with_usage(), out)
        assert out == []

    def test_does_not_emit_usage_without_input_tokens(self) -> None:
        from opendatasci.streaming.events import TokenEvent, UsageEvent

        p = self._proc()
        out = [TokenEvent(content="hi")]
        p._update_stream_usage(self._chunk_with_usage(), out)
        assert not any(isinstance(e, UsageEvent) for e in out)

    def test_output_chars_accumulate_across_calls(self) -> None:
        from opendatasci.streaming.events import TokenEvent, UsageEvent

        p = self._proc()
        p._stream_input_tokens = 100
        out1 = [TokenEvent(content="hello")]
        p._update_stream_usage(self._chunk_with_usage(), out1)
        tokens_after_first = next(e for e in out1 if isinstance(e, UsageEvent)).output_tokens

        out2 = [TokenEvent(content=" world!")]
        p._update_stream_usage(self._chunk_with_usage(), out2)
        tokens_after_second = next(e for e in out2 if isinstance(e, UsageEvent)).output_tokens

        assert tokens_after_second >= tokens_after_first

    def test_output_tokens_at_least_one(self) -> None:
        from opendatasci.streaming.events import TokenEvent, UsageEvent

        p = self._proc()
        p._stream_input_tokens = 100
        out = [TokenEvent(content="x")]
        p._update_stream_usage(self._chunk_with_usage(), out)
        usage = next(e for e in out if isinstance(e, UsageEvent))
        assert usage.output_tokens >= 1
