import logging
import uuid
from abc import ABC, abstractmethod
from contextlib import AsyncExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    RemoveMessage,
    SystemMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from opendatasci._utils.graph_utils import is_interrupt_state_snapshot
from opendatasci._utils.message_utils import (
    get_final_ai_message,
    get_message_text_content,
    is_final_ai_message,
    to_text_content_blocks,
)
from opendatasci._utils.pydantic_utils import FrozenStrictBaseModel
from opendatasci._utils.streaming_utils import format_stream_error
from opendatasci.agents.chat_history import ChatHistoryBuilder
from opendatasci.agents.graphs import AgentCompiledGraph, AgentGraphFactory
from opendatasci.agents.interrupts import InterruptKind
from opendatasci.agents.states import AgentState
from opendatasci.configs import OpenDataSciConfig
from opendatasci.context.base import BaseContextStore
from opendatasci.context.local import LocalContextStore
from opendatasci.memory.chat_memory import ChatHistoryCompactor
from opendatasci.memory.messages import MessageOrigin, TaskMessage, UserMessage
from opendatasci.memory.turn_memory import TurnRewinder
from opendatasci.models.factory import (
    _RetryRunnable,
    create_model,
    create_secondary_model,
    with_retry,
)
from opendatasci.prompts.builders import SystemContextBuilder
from opendatasci.sandbox.base import BaseSandbox, BaseSandboxFactory
from opendatasci.sandbox.srt import SRTSandboxFactory
from opendatasci.session import BaseSessionManager, LocalSessionManager
from opendatasci.skills import BaseSkillStore, LocalSkillStore
from opendatasci.streaming import (
    AgentStreamEvent,
    AgentTurnStreamProcessor,
    ApprovalRequiredEvent,
    ErrorEvent,
    InputRequiredEvent,
    MessageEvent,
    ResponseEvent,
)
from opendatasci.tasks.base import AgentTaskManagerBase
from opendatasci.tasks.local import LocalAgentTaskManager
from opendatasci.tools.factory import (
    create_execution_mode_tools,
    create_plan_mode_tools,
    create_self_review_mode_tools,
)
from opendatasci.workspace.base import BaseWorkspace

logger = logging.getLogger(__name__)

AGENT_RECURSION_LIMIT: int = 1000


class Invocation(FrozenStrictBaseModel):
    """One message to fold into a single turn, alongside others.

    ``origin`` marks whether this is a background task's output
    (:attr:`MessageOrigin.TASK`) or user-typed text (:attr:`MessageOrigin.USER`);
    the two are rendered as different message types so the model can tell
    them apart. ``content`` follows the same content-block shape as
    :attr:`~langchain_core.messages.BaseMessage.content` (a list of
    ``{"type": ..., ...}`` blocks), not a plain string.
    """

    content: list[dict[str, Any]]
    created_at: datetime
    origin: MessageOrigin = MessageOrigin.USER

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        origin: MessageOrigin = MessageOrigin.USER,
        created_at: datetime | None = None,
    ) -> "Invocation":
        """Build an Invocation from plain text."""
        return cls(
            content=to_text_content_blocks(text),
            created_at=created_at if created_at is not None else datetime.now(timezone.utc),
            origin=origin,
        )


class BaseOpenDataSciAgent(ABC):
    @abstractmethod
    def astream(
        self, invocation: Invocation | list[Invocation]
    ) -> AsyncIterator[AgentStreamEvent]: ...

    @abstractmethod
    def resume_with_input(self, answer: str) -> AsyncIterator[AgentStreamEvent]: ...

    @abstractmethod
    def resume_with_approval(self, approved: bool) -> AsyncIterator[AgentStreamEvent]: ...

    @abstractmethod
    def is_user_input_required(self) -> bool: ...

    @abstractmethod
    async def rewind_turn(self) -> None: ...

    @abstractmethod
    async def clear_chat_history(self) -> None: ...

    @abstractmethod
    async def compact_chat_history(self) -> str: ...

    @property
    @abstractmethod
    def task_manager(self) -> AgentTaskManagerBase: ...


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
        agent_task_manager: Tracks background tasks spawned via the ``task``
            tool.  Defaults to a file-backed :class:`LocalAgentTaskManager`.
        session_manager: Tracks the session's conversation threads in the
            graph checkpointer; clearing the conversation creates a new
            thread.  Defaults to a file-backed :class:`LocalSessionManager`.
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
        agent_task_manager: AgentTaskManagerBase | None = None,
        sandbox_factory: BaseSandboxFactory | None = None,
        checkpointer: BaseCheckpointSaver[Any] | None = None,
        tools: list[BaseTool] | None = None,
        session_id: str | None = None,
        session_manager: BaseSessionManager | None = None,
        config: OpenDataSciConfig | None = None,
    ) -> None:
        self._workspace = workspace
        self._session_id = session_id or uuid.uuid4().hex
        self._config = (config or OpenDataSciConfig()).model_copy(deep=True)
        self._tools = tools
        self._sandbox_factory = sandbox_factory
        self._skill_store = skill_store
        self._agent_task_manager = agent_task_manager
        self._context_store = context_store
        self._session_manager = session_manager
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
        if self._agent_task_manager is None:
            output_root = Path(self._context_store.root) / "workers" / "outputs"
            self._agent_task_manager = LocalAgentTaskManager(output_root=output_root)
        if self._session_manager is None:
            self._session_manager = LocalSessionManager(
                workspace_path=Path(self._workspace.get_reference()),
                session_id=self._session_id,
            )
        checkpointer = self._checkpointer or MemorySaver()

        self._llm: BaseChatModel = create_model(self._config)
        self._summarizer_llm: BaseChatModel = create_secondary_model(self._config)

        self._sandbox: BaseSandbox = await self._exit_stack.enter_async_context(
            self._sandbox_factory.create(workspace_path=Path(self._workspace.get_reference()))
        )

        if self._tools is None:
            self._tools = create_execution_mode_tools(
                self._workspace,
                self._sandbox,
                self._context_store,
                self._sandbox_factory,
                session_id=self._session_id,
                skill_store=self._skill_store,
                datasci_config=self._config,
                agent_task_manager=self._agent_task_manager,
            )

        self._tools_in_plan_mode: list[BaseTool] = create_plan_mode_tools(self._tools)
        self._tools_in_self_review_mode: list[BaseTool] = create_self_review_mode_tools(self._tools)

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

        self._graph: AgentCompiledGraph = self._build_graph(checkpointer)
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self._exit_stack.aclose()

    @property
    def task_manager(self) -> AgentTaskManagerBase:
        """The task manager tracking this agent's background (``task``, ``run_mode="background"``) work.

        Exposed so a caller driving this agent (the TUI, or a hosted-service
        equivalent) can watch for background-task completions via
        :meth:`AgentTaskManagerBase.listen_task_updates`.
        """
        return self._agent_task_manager  # type: ignore[return-value]

    @property
    def _graph_config(self) -> RunnableConfig:
        thread_id = self._session_manager.get_or_create_thread()  # type: ignore[union-attr]
        return {"configurable": {"thread_id": str(thread_id)}}

    @property
    def graph(self) -> AgentCompiledGraph:
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
            active_skill_domain=(
                state.active_skill_domains[0] if state.active_skill_domains else None
            ),
            is_plan_mode=state.is_plan_mode,
            is_self_review_mode=state.is_self_review_mode,
        )

    def _build_graph(self, checkpointer: BaseCheckpointSaver[Any] | None) -> AgentCompiledGraph:
        return AgentGraphFactory(
            get_llm_with_tools=self._get_active_llm_with_tools,
            tools=self._tools,  # type: ignore[arg-type]
            build_system_context=self._build_system_context,
            chat_history_builder=self._chat_history_builder,
            checkpointer=checkpointer,
        ).build()

    @classmethod
    def _prepare_user_message(cls, query: str) -> UserMessage:
        return UserMessage(
            content=to_text_content_blocks(query), created_at=datetime.now(timezone.utc)
        )

    @classmethod
    def _prepare_batch_messages(cls, items: list[Invocation]) -> list[BaseMessage]:
        """Build one message per item, worker results first, then user text."""
        worker_items = [item for item in items if item.origin == MessageOrigin.TASK]
        user_items = [item for item in items if item.origin == MessageOrigin.USER]
        return [
            *(
                TaskMessage(content=item.content, created_at=item.created_at)
                for item in worker_items
            ),
            *(UserMessage(content=item.content, created_at=item.created_at) for item in user_items),
        ]

    def _thread_config(self, thread_id: Any) -> RunnableConfig:
        return {
            "recursion_limit": AGENT_RECURSION_LIMIT,
            "configurable": {"thread_id": str(thread_id)},
        }

    @staticmethod
    def _pending_interrupt_value(graph_state: Any) -> dict[str, Any] | None:
        if not is_interrupt_state_snapshot(graph_state):
            return None
        return graph_state.tasks[0].interrupts[0].value  # type: ignore[no-any-return]

    @staticmethod
    def _is_approval_interrupt(intr_value: dict[str, Any]) -> bool:
        return isinstance(intr_value, dict) and intr_value.get("kind") == InterruptKind.APPROVAL

    def _require_pending_interrupt(self, *, approval: bool) -> None:
        intr_value = self._pending_interrupt_value(self._graph.get_state(self._graph_config))
        if intr_value is None:
            raise RuntimeError(
                "no interrupt is currently pending; call astream() to run a turn instead"
            )
        is_approval = self._is_approval_interrupt(intr_value)
        if is_approval != approval:
            expected = "resume_with_approval()" if is_approval else "resume_with_input()"
            raise RuntimeError(f"a different kind of input is pending; call {expected} instead")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def astream(
        self, invocation: Invocation | list[Invocation]
    ) -> AsyncIterator[AgentStreamEvent]:
        """Stream a response to *invocation*, yielding :class:`AgentStreamEvent` objects.

        Starts (or continues) a turn. A list of :class:`Invocation` is **not**
        several separate requests — it is one request whose items are all folded
        into a single turn, producing exactly one response.

        Raises :class:`RuntimeError` if the agent is currently paused awaiting a
        response to an :class:`~opendatasci.streaming.InputRequiredEvent` or
        :class:`~opendatasci.streaming.ApprovalRequiredEvent` — resume it with
        :meth:`resume_with_input` or :meth:`resume_with_approval` instead.
        """
        if self.is_user_input_required():
            raise RuntimeError(
                "the agent is awaiting a response to a pending question or approval "
                "request; call resume_with_input() or resume_with_approval() instead"
            )
        thread_id = self._session_manager.get_or_create_thread()  # type: ignore[union-attr]
        config = self._thread_config(thread_id)
        items = invocation if isinstance(invocation, list) else [invocation]

        self._context_store.prune()  # type: ignore[union-attr]
        graph_input: Any = {
            "messages": type(self)._prepare_batch_messages(items),
            "active_skills": [],
            "active_skill_domains": [],
            "is_plan_mode": False,
            "is_self_review_mode": False,
        }

        async for event in self._stream_turn(graph_input, thread_id, config):
            yield event

    async def resume_with_input(self, answer: str) -> AsyncIterator[AgentStreamEvent]:
        """Answer a pending :class:`~opendatasci.streaming.InputRequiredEvent` with *answer*.

        Raises :class:`RuntimeError` if the agent isn't currently paused on a
        free-text/choice question.
        """
        self._require_pending_interrupt(approval=False)
        thread_id = self._session_manager.get_or_create_thread()  # type: ignore[union-attr]
        config = self._thread_config(thread_id)
        async for event in self._stream_turn(Command(resume=answer), thread_id, config):
            yield event

    async def resume_with_approval(self, approved: bool) -> AsyncIterator[AgentStreamEvent]:
        """Answer a pending :class:`~opendatasci.streaming.ApprovalRequiredEvent` with *approved*.

        Raises :class:`RuntimeError` if the agent isn't currently paused on a
        command-approval request.
        """
        self._require_pending_interrupt(approval=True)
        thread_id = self._session_manager.get_or_create_thread()  # type: ignore[union-attr]
        config = self._thread_config(thread_id)
        async for event in self._stream_turn(Command(resume=approved), thread_id, config):
            yield event

    def is_user_input_required(self) -> bool:
        """True iff the agent is paused awaiting the user's answer to a pending question or approval request."""
        return is_interrupt_state_snapshot(self._graph.get_state(self._graph_config))

    async def _stream_turn(
        self, graph_input: Any, thread_id: Any, config: RunnableConfig
    ) -> AsyncIterator[AgentStreamEvent]:
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
        intr_value = self._pending_interrupt_value(graph_state)
        if intr_value is not None:
            if self._is_approval_interrupt(intr_value):
                yield ApprovalRequiredEvent(
                    command=intr_value["command"],
                    description=intr_value["description"],
                    heads_up=intr_value["heads_up"],
                )
            else:
                yield InputRequiredEvent(
                    content=intr_value["question"],
                    choices=intr_value["choices"],
                )
            return

        completed_turn_messages = graph_state.values["messages"]
        final_ai_msg = get_final_ai_message(completed_turn_messages)
        final_response = get_message_text_content(final_ai_msg).strip()

        # A clear_chat_history() issued while this turn was streaming created
        # a new thread; summarizing the cleared turn would leak it back in.
        if thread_id == self._session_manager.get_current_thread():  # type: ignore[union-attr]
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
        """Clear all conversation context (preserves session state such as the sandbox).

        Drops the conversation history, turn summaries, compaction, active
        skills, mode flags, and any pending interrupt by starting a fresh
        checkpointer thread; cancels any in-flight turn summarization; and
        deletes the session's persisted plan so it is no longer recalled.
        """
        self._chat_history_builder.cancel_pending_tasks()
        self._session_manager.create_thread()  # type: ignore[union-attr]
        if self._context_store is not None:
            self._context_store.clear_plans(self._session_id)

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
