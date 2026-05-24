import logging
import uuid
from abc import ABC, abstractmethod
from contextlib import AsyncExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from opendatasci._utils.langchain_utils import (
    get_final_ai_message,
    get_message_text_content,
    is_interrupt_state_snapshot,
)
from opendatasci._utils.streaming_utils import format_stream_error
from opendatasci.agents.chat_memory import (
    ChatHistoryBuilder,
    ChatHistoryCompactor,
    TurnSummarizer,
    extract_thinking_and_text,
)
from opendatasci.agents.graphs import AgentGraphFactory, WorkerGraphFactory
from opendatasci.agents.states import AgentState
from opendatasci.agents.turn_memory import AgentLoopCompactor, TurnRewinder
from opendatasci.configs import OpenDataSciConfig
from opendatasci.context.base import BaseContextStore
from opendatasci.context.local import LocalContextStore
from opendatasci.models.factory import (
    _RetryRunnable,
    create_model,
    create_secondary_model,
    with_retry,
)
from opendatasci.prompts.builders import SystemContextBuilder
from opendatasci.prompts.caching import cached_system_prompt
from opendatasci.sandbox.base import BaseSandbox, BaseSandboxFactory
from opendatasci.sandbox.srt import SRTSandboxFactory
from opendatasci.skills import BaseSkillStore, LocalSkillStore
from opendatasci.skills.base import Skill
from opendatasci.streaming import (
    AgentStreamEvent,
    AgentTurnStreamProcessor,
    ErrorEvent,
    InputRequiredEvent,
    MessageEvent,
    ResponseEvent,
)
from opendatasci.tools import (
    ToolName,
    create_agent_tools,
)
from opendatasci.workspace.base import BaseWorkspace

logger = logging.getLogger(__name__)

AGENT_RECURSION_LIMIT: int = 1000

SUBAGENT_TAG: str = "opendatasci:subagent"
WORKER_MAX_STEPS: int = 50

# Signature: (event_type, content, metadata | None) -> None
OnEventCallback = Callable[[str, str, "dict[str, Any] | None"], None]

_ARGS_PREVIEW_LEN = 80

__all__ = [
    "Agent",
    "ParallelWorkerAgent",
    "SUBAGENT_TAG",
    "WORKER_MAX_STEPS",
    "OnEventCallback",
    "extract_thinking_and_text",
]


class BaseOpenDataSciAgent(ABC):
    """Abstract interface for the data science agent."""

    @abstractmethod
    def astream(self, query: str) -> AsyncIterator[AgentStreamEvent]: ...

    @abstractmethod
    async def rewind_turn(self) -> None: ...

    @abstractmethod
    async def clear_chat_history(self) -> None: ...

    @abstractmethod
    async def compact_chat_history(self) -> str: ...


class Agent(BaseOpenDataSciAgent):
    """Data science and machine learning conversational AI agent.

    Must be used as an async context manager; the sandbox is created on entry
    and closed on exit::

        async with Agent(...) as agent:
            async for event in agent.astream("analyse the data"):
                ...

    For most use cases prefer the :func:`create_agent` factory, which wires
    all dependencies from a file or directory path.

    Args:
        workspace: The workspace the agent operates on.
        session_id: Identifier for this session.  Generated automatically
            when omitted.
        context_store: Store that supplies dataset profiles and notes for the
            active workspace and persists the agent's plan across turns.  A
            local file-based store is created when omitted.
        skill_store: Registry that the agent queries to resolve named skills
            at runtime.  Defaults to the built-in :class:`LocalSkillStore`.
        sandbox_factory: Factory used to create the execution sandbox.
            The sandbox lifetime is tied to the agent's context manager scope.
            Defaults to :class:`SRTSandboxFactory`.
        checkpointer: Checkpoint backend for graph state.  Defaults to an
            in-memory store.
        tools: Full set of tools available to the agent.  Plan mode and
            self-review mode use this list minus worker-spawning tools.
            Override to restrict capabilities or inject custom tools.
        config: LLM provider and model settings.  Defaults to
            :class:`OpenDataSciConfig` with its built-in defaults.
    """

    def __init__(
        self,
        workspace: BaseWorkspace,
        context_store: BaseContextStore | None = None,
        skill_store: BaseSkillStore | None = None,
        sandbox_factory: BaseSandboxFactory | None = None,
        checkpointer: BaseCheckpointSaver[Any] | None = None,
        tools: list[BaseTool] | None = None,
        session_id: str | None = None,
        config: OpenDataSciConfig | None = None,
    ) -> None:
        self._workspace = workspace
        self._session_id = session_id or uuid.uuid4().hex
        self._config = (config or OpenDataSciConfig()).model_copy(deep=True)
        self._tools = tools
        self._sandbox_factory = sandbox_factory
        self._skill_store = skill_store
        self._context_store = context_store
        self._checkpointer = checkpointer

    async def __aenter__(self) -> "Agent":
        self._exit_stack = AsyncExitStack()

        if self._sandbox_factory is None:
            self._sandbox_factory = SRTSandboxFactory(
                command_timeout=self._config.local_code_exec_timeout
            )
        if self._skill_store is None:
            self._skill_store = LocalSkillStore()
        if self._context_store is None:
            workspace_path = Path(self._workspace.get_reference())
            self._context_store = LocalContextStore(workspace_path=workspace_path)
        checkpointer = self._checkpointer or MemorySaver()

        self._llm: BaseChatModel = create_model(self._config)
        self._summarizer_llm: BaseChatModel = create_secondary_model(self._config)

        self._sandbox: BaseSandbox = await self._exit_stack.enter_async_context(
            self._sandbox_factory.create(workspace_path=Path(self._workspace.get_reference()))
        )

        if self._tools is None:
            self._tools = create_agent_tools(
                self._workspace,
                self._sandbox,
                self._context_store,
                self._sandbox_factory,
                store=self._skill_store,
                datasci_config=self._config,
                save_plan=lambda plan: self._context_store.save_plan(self._session_id, plan),  # type: ignore[union-attr]
            )

        tools_restricted = [t for t in self._tools if t.name != ToolName.SPAWN_WORKERS]
        self._tools_in_plan_mode: list[BaseTool] = tools_restricted
        self._tools_in_self_review_mode: list[BaseTool] = tools_restricted

        self._llm_with_tools: _RetryRunnable = with_retry(self._llm.bind_tools(self._tools))
        self._llm_with_tools_plan: _RetryRunnable = with_retry(
            self._llm.bind_tools(self._tools_in_plan_mode)
        )
        self._llm_with_tools_self_review: _RetryRunnable = with_retry(
            self._llm.bind_tools(self._tools_in_self_review_mode)
        )

        self._system_context_builder = SystemContextBuilder(
            config=self._config,
            context_store=self._context_store,
            session_id=self._session_id,
        )
        summarizer = TurnSummarizer(summarizer_llm=self._summarizer_llm)
        loop_compactor = AgentLoopCompactor(llm=self._llm)
        self._chat_history_builder = ChatHistoryBuilder(
            summarizer=summarizer,
            loop_compactor=loop_compactor,
            midturn_compaction_threshold=self._config.midturn_compaction_threshold,
        )

        self._graph: CompiledStateGraph = self._build_graph(checkpointer)
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self._exit_stack.aclose()

    @property
    def _graph_config(self) -> RunnableConfig:
        return {"configurable": {"thread_id": self._session_id}}

    @property
    def graph(self) -> CompiledStateGraph:
        """Return the underlying compiled state graph."""
        return self._graph

    def _get_active_llm_with_tools(self, state: AgentState) -> _RetryRunnable:
        """Return the LLM binding that matches the current agent mode."""
        if state.is_self_review_mode:
            return self._llm_with_tools_self_review
        if state.is_plan_mode:
            return self._llm_with_tools_plan
        return self._llm_with_tools

    def _build_system_context(
        self, state: AgentState, memory_text: str | None
    ) -> list[SystemMessage]:
        return self._system_context_builder.build(
            active_skills=state.active_skills,
            is_plan_mode=state.is_plan_mode,
            is_self_review_mode=state.is_self_review_mode,
            memory_text=memory_text,
        )

    def _build_graph(self, checkpointer: BaseCheckpointSaver[Any] | None) -> CompiledStateGraph:
        return AgentGraphFactory(
            get_llm_with_tools=self._get_active_llm_with_tools,
            tools=self._tools,  # type: ignore[arg-type]
            build_system_context=self._build_system_context,
            chat_history_builder=self._chat_history_builder,
            checkpointer=checkpointer,
        ).build()

    @classmethod
    def _prepare_user_message(cls, query: str) -> HumanMessage:
        """Build the turn-opening HumanMessage.

        The start timestamp and an ``is_input_on_interrupt`` flag are stored in
        ``additional_kwargs`` so the turn's start time and boundary can be
        recovered later from the message history (see ``get_last_turn_messages``),
        instead of being held as per-turn agent state.
        """
        return HumanMessage(
            content=query,
            additional_kwargs={
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "is_input_on_interrupt": False,
            },
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def astream(self, user_input: str) -> AsyncIterator[AgentStreamEvent]:
        """Stream a response to *user_input*, yielding ``AgentStreamEvent`` objects.

        If the previous call ended with an ``input_required`` event, pass the
        user's answer here to resume the interrupted graph run.  Otherwise
        *user_input* is treated as a new query.
        """
        config: RunnableConfig = {
            "recursion_limit": AGENT_RECURSION_LIMIT,
            "configurable": {"thread_id": self._session_id},
        }

        graph_state = self._graph.get_state(config)
        if is_interrupt_state_snapshot(graph_state):
            graph_input: Any = Command(resume=user_input)
        else:
            self._context_store.prune()  # type: ignore[union-attr]
            user_msg = type(self)._prepare_user_message(user_input)
            graph_input = {
                "messages": [user_msg],
                "active_skills": [],
                "is_plan_mode": False,
                "is_self_review_mode": False,
            }

        processor = AgentTurnStreamProcessor()

        try:
            async for event in self._graph.astream_events(graph_input, version="v2", config=config):
                for stream_event in processor.process_event(event):  # type: ignore[arg-type]
                    if not isinstance(stream_event, MessageEvent):
                        yield stream_event
        except Exception as exc:
            yield ErrorEvent(content=format_stream_error(exc))
            return

        graph_state = self._graph.get_state(config)
        if is_interrupt_state_snapshot(graph_state):
            intr_value = graph_state.tasks[0].interrupts[0].value
            yield InputRequiredEvent(
                content=intr_value["question"],
                choices=intr_value["choices"],
            )
            return

        messages = graph_state.values["messages"]
        final_ai_msg = get_final_ai_message(messages)
        final_response = get_message_text_content(final_ai_msg).strip()

        self._chat_history_builder.schedule_turn_summarization(messages)

        yield ResponseEvent(content=final_response)

    async def rewind_turn(self) -> None:
        """Remove the last turn from the conversation history."""
        snapshot = await self._graph.aget_state(self._graph_config)
        messages = snapshot.values.get("messages", [])
        if not messages:
            return
        self._chat_history_builder.cancel_pending()
        rewinder = TurnRewinder()
        new_messages = rewinder.rewind_last_turn(messages)
        removed = messages[len(new_messages) :]
        if removed:
            self._graph.update_state(
                self._graph_config,
                {"messages": [RemoveMessage(id=msg.id) for msg in removed]},
            )

    async def clear_chat_history(self) -> None:
        """Clear conversation history and rolling memory (preserves session state)."""
        snapshot = await self._graph.aget_state(self._graph_config)
        messages = snapshot.values.get("messages", [])
        self._chat_history_builder.cancel_pending()
        updates: dict[str, Any] = {"turn_summaries": [], "session_preamble": None}
        if messages:
            updates["messages"] = [RemoveMessage(id=msg.id) for msg in messages]
        self._graph.update_state(self._graph_config, updates)

    async def compact_chat_history(self) -> str:
        """Compact the conversation history using the LLM.

        Older turns are summarised and discarded; the most recent turn is kept
        verbatim.  Returns the summary text, or a placeholder when there is not
        enough history to compact.
        """
        snapshot = self._graph.get_state(self._graph_config)
        messages = snapshot.values.get("messages", [])
        if not messages:
            return "(no conversation to compact)"

        compactor = ChatHistoryCompactor(self._llm)
        try:
            new_messages = await compactor.compact(messages)
        except ValueError:
            return "(no conversation to compact)"

        if len(new_messages) == len(messages):
            return "(no conversation to compact)"

        # Extract the raw LLM summary from the leading compaction SystemMessage, stripping
        # the <compacted_history> wrapper added by ChatHistoryCompactor, then discard the
        # SystemMessage so it never accumulates in graph state.
        compaction_msg = new_messages[0]
        raw = (
            compaction_msg.content
            if isinstance(compaction_msg.content, str)
            else str(compaction_msg.content)
        )
        _prefix, _suffix = "<compacted_history>\n", "\n</compacted_history>"
        summary = (
            raw[len(_prefix) : -len(_suffix)]
            if raw.startswith(_prefix) and raw.endswith(_suffix)
            else raw
        )
        kept_messages = [m for m in new_messages if not isinstance(m, SystemMessage)]

        # The compacted summary becomes the session preamble so the agent retains
        # the older context, and the rolling per-turn summaries (now folded into
        # that summary) are reset.
        self._chat_history_builder.cancel_pending()
        message_updates: list[Any] = [RemoveMessage(id=msg.id) for msg in messages] + kept_messages
        self._graph.update_state(
            self._graph_config,
            {
                "messages": message_updates,
                "session_preamble": summary,
                "turn_summaries": [],
            },
        )
        return summary


class ParallelWorkerAgent:
    """One-shot worker agent that executes a single delegated subtask to completion."""

    def __init__(
        self,
        tools: list[BaseTool],
        config: OpenDataSciConfig | None = None,
        llm: BaseChatModel | None = None,
    ) -> None:
        self._config = config or OpenDataSciConfig()
        _llm = llm if llm is not None else create_model(self._config)
        _llm_with_tools = with_retry(_llm.bind_tools(tools))
        self._current_system_prompt: str = ""

        self._graph = WorkerGraphFactory(
            llm_with_tools=_llm_with_tools,
            tools=tools,
            build_system_context=self._build_system_context,
        ).build()

    def _build_system_context(
        self, state: AgentState, memory_text: str | None
    ) -> list[SystemMessage]:
        messages: list[SystemMessage] = [
            SystemMessage(
                content=cached_system_prompt(self._current_system_prompt, self._config.provider)  # type: ignore[arg-type]
            )
        ]
        for skill in state.active_skills:
            messages.append(
                SystemMessage(
                    content=cached_system_prompt(skill.content, self._config.provider)  # type: ignore[arg-type]
                )
            )
        return messages

    async def ainvoke(
        self,
        task: str,
        system_prompt: str,
        on_event: OnEventCallback | None = None,
        messages_out: "list[Any] | None" = None,
        initial_active_skills: "list[Skill] | None" = None,
    ) -> str:
        """Execute *task* to completion and return the final text response."""
        self._current_system_prompt = system_prompt
        initial_state = AgentState(
            messages=[HumanMessage(content=task)],
            active_skills=list(initial_active_skills or []),
        )
        invoke_config: RunnableConfig = {
            "tags": [SUBAGENT_TAG],
            "recursion_limit": WORKER_MAX_STEPS * 2 + 1,
        }

        final_state: dict[str, Any] | None = None

        if on_event is not None:
            async for event in self._graph.astream_events(
                initial_state, version="v2", config=invoke_config
            ):
                kind = event["event"]
                if kind == "on_tool_start":
                    tool_name = event["name"]
                    args = event["data"].get("input") or {}
                    args_preview = str(args)[:_ARGS_PREVIEW_LEN]
                    summary = args.get("summary", "") if isinstance(args, dict) else ""
                    on_event(
                        "worker_tool_call",
                        tool_name,
                        {"args_preview": args_preview, "summary": summary},
                    )
                elif kind == "on_tool_end":
                    tool_name = event["name"]
                    output = event["data"].get("output")
                    if isinstance(output, ToolMessage):
                        content = output.content
                    elif isinstance(output, str):
                        content = output
                    else:
                        content = ""
                    is_error = isinstance(content, str) and content.startswith("Error")
                    on_event("worker_tool_result", tool_name, {"success": not is_error})
                elif kind == "on_chain_end" and event.get("name") == "LangGraph":
                    final_state = event["data"].get("output")
        else:
            final_state = await self._graph.ainvoke(initial_state, config=invoke_config)

        if messages_out is not None and final_state is not None:
            final_messages = final_state.get("messages", [])
            final_active_skills: list[Skill] = final_state.get("active_skills", [])
            dummy_state = AgentState(messages=[], active_skills=final_active_skills)
            sys_messages = self._build_system_context(dummy_state, None)
            messages_out.extend([*sys_messages, *final_messages])

        if final_state is None:
            raise RuntimeError("Worker graph ended without producing output")

        messages = final_state.get("messages", [])
        if not messages:
            raise RuntimeError("Worker graph ended with no messages")

        last = messages[-1]
        return get_message_text_content(last).strip()
