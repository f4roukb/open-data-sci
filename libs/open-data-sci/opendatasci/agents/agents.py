import logging
import uuid
from abc import ABC, abstractmethod
from contextlib import AsyncExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
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

from opendatasci._utils.graph_utils import is_interrupt_state_snapshot
from opendatasci._utils.message_utils import (
    get_final_ai_message,
    get_message_text_content,
    is_final_ai_message,
)
from opendatasci._utils.streaming_utils import format_stream_error
from opendatasci.agents.chat_history import ChatHistoryBuilder
from opendatasci.agents.graphs import AgentGraphFactory, WorkerGraphFactory
from opendatasci.agents.states import AgentState
from opendatasci.configs import OpenDataSciConfig
from opendatasci.context.base import BaseContextStore
from opendatasci.context.local import LocalContextStore
from opendatasci.memory.chat_memory import ChatHistoryCompactor
from opendatasci.memory.messages import HarnessMessage, UserMessage
from opendatasci.memory.turn_memory import TurnRewinder
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


class BaseOpenDataSciAgent(ABC):
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
                session_id=self._session_id,
                store=self._skill_store,
                datasci_config=self._config,
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

        self._system_context_builder = SystemContextBuilder(config=self._config)
        self._chat_history_builder = ChatHistoryBuilder(
            summarizer_llm=self._summarizer_llm,
            loop_compactor_llm=self._llm,
            midturn_compaction_threshold=self._config.midturn_compaction_threshold,
            context_store=self._context_store,
            session_id=self._session_id,
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
        return self._graph

    def _get_active_llm_with_tools(self, state: AgentState) -> _RetryRunnable:
        if state.is_self_review_mode:
            return self._llm_with_tools_self_review
        if state.is_plan_mode:
            return self._llm_with_tools_plan
        return self._llm_with_tools

    def _build_system_context(self, state: AgentState) -> list[SystemMessage]:
        return self._system_context_builder.build(
            active_skills=state.active_skills,
            is_plan_mode=state.is_plan_mode,
            is_self_review_mode=state.is_self_review_mode,
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
    def _prepare_user_message(cls, query: str) -> UserMessage:
        return UserMessage(content=query, created_at=datetime.now(timezone.utc))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def astream(self, user_input: str) -> AsyncIterator[AgentStreamEvent]:
        """Stream a response to *user_input*, yielding :class:`AgentStreamEvent` objects.

        If the previous call ended with an :class:`~opendatasci.streaming.InputRequiredEvent`,
        pass the user's answer here to resume; otherwise *user_input* starts a new turn.
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

        completed_turn_messages = graph_state.values["messages"]
        final_ai_msg = get_final_ai_message(completed_turn_messages)
        final_response = get_message_text_content(final_ai_msg).strip()

        self._chat_history_builder.schedule_turn_summarization(completed_turn_messages)

        yield ResponseEvent(content=final_response)

    async def rewind_turn(self) -> None:
        """Remove the last turn from the conversation history."""
        snapshot = await self._graph.aget_state(self._graph_config)
        ongoing_turn_messages = snapshot.values.get("messages", [])
        if not ongoing_turn_messages:
            return
        self._chat_history_builder.cancel_pending_tasks()
        rewinder = TurnRewinder()
        kept_messages = rewinder.rewind_last_turn(ongoing_turn_messages)
        removed = ongoing_turn_messages[len(kept_messages) :]
        if removed:
            self._graph.update_state(
                self._graph_config,
                {"messages": [RemoveMessage(id=msg.id) for msg in removed]},
            )

    async def clear_chat_history(self) -> None:
        """Clear conversation history and rolling memory (preserves session state)."""
        snapshot = await self._graph.aget_state(self._graph_config)
        ongoing_turn_messages = snapshot.values.get("messages", [])
        self._chat_history_builder.cancel_pending_tasks()
        updates: dict[str, Any] = {"turn_summaries": [], "chat_history_compaction": None}
        if ongoing_turn_messages:
            updates["messages"] = [RemoveMessage(id=msg.id) for msg in ongoing_turn_messages]
        self._graph.update_state(self._graph_config, updates)

    async def compact_chat_history(self) -> str:
        """Fold the rolling turn summaries into a single compaction summary.

        Includes any existing compaction, all turn summaries, and the current
        completed turn (if any) in the compaction context. Clears turn summaries
        and replaces any existing compaction with the new one. An ongoing (incomplete)
        turn is left untouched.

        Returns the compaction text, or a placeholder when there is nothing to compact.
        """
        snapshot = self._graph.get_state(self._graph_config)
        turn_summaries = snapshot.values.get("turn_summaries", [])
        existing_compaction = snapshot.values.get("chat_history_compaction", None)
        current_messages = snapshot.values.get("messages", [])

        # Include the current turn only when it is complete.
        completed_messages = (
            current_messages
            if current_messages and is_final_ai_message(current_messages[-1])
            else []
        )

        compactor = ChatHistoryCompactor(self._llm)
        try:
            compaction_summary = await compactor.compact(
                existing_compaction=existing_compaction,
                turn_summaries=turn_summaries,
                completed_messages=completed_messages,
            )
        except ValueError:
            return "(no conversation to compact)"

        self._chat_history_builder.cancel_pending_tasks()
        updates: dict[str, Any] = {
            "turn_summaries": [],
            "chat_history_compaction": compaction_summary,
        }
        if completed_messages:
            updates["messages"] = [RemoveMessage(id=msg.id) for msg in completed_messages]
        self._graph.update_state(self._graph_config, updates)
        return compaction_summary.content


class ConcurrentWorkerAgent:
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

    def _build_system_context(self, state: AgentState) -> list[SystemMessage]:
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
            messages=[HarnessMessage(content=task)],
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
            sys_messages = self._build_system_context(dummy_state)
            messages_out.extend([*sys_messages, *final_messages])

        if final_state is None:
            raise RuntimeError("Worker graph ended without producing output")

        messages = final_state.get("messages", [])
        if not messages:
            raise RuntimeError("Worker graph ended with no messages")

        last = messages[-1]
        return get_message_text_content(last).strip()
