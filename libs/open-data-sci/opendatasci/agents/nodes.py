from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, Optional

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig

from opendatasci._utils.mixins import RenderableMessageMixin
from opendatasci.agents.chat_history import ChatHistoryBuilder
from opendatasci.agents.states import AgentState
from opendatasci.memory.messages import AgentMessage, TaskMessage
from opendatasci.models.factory import _RetryRunnable
from opendatasci.tasks.base import AgentTaskManagerBase, AgentTaskRecord

BuildSystemContext = Callable[[AgentState], list[SystemMessage]]


class BaseNode(ABC):
    """Base class for all agent graph nodes.

    Subclasses implement ``ainvoke()`` as the primary async entry-point.
    ``to_async_callable()`` wraps it as an async callable for use in a graph.
    """

    @abstractmethod
    async def ainvoke(
        self, state: AgentState, config: Optional[RunnableConfig] = None
    ) -> dict[str, Any]:
        """Async entry-point; must return a partial state dict."""
        ...

    def to_async_callable(
        self,
    ) -> Callable[..., Awaitable[dict[str, Any]]]:
        """Return an async callable that delegates to ``ainvoke()``."""

        async def node_fn(
            state: AgentState, config: Optional[RunnableConfig] = None
        ) -> dict[str, Any]:
            return await self.ainvoke(state, config)

        return node_fn


class AgentNode(BaseNode):
    """Graph node that invokes the LLM and returns the updated message list."""

    def __init__(
        self,
        get_llm_with_tools: Callable[[AgentState], _RetryRunnable],
        build_system_context: BuildSystemContext,
        chat_history_builder: ChatHistoryBuilder | None = None,
    ) -> None:
        self._get_llm_with_tools = get_llm_with_tools
        self._build_system_context = build_system_context
        self._chat_history_builder = chat_history_builder

    async def ainvoke(
        self, state: AgentState, config: Optional[RunnableConfig] = None
    ) -> dict[str, Any]:
        updates: dict[str, Any] = {}

        if self._chat_history_builder is not None:
            turn_context = await self._chat_history_builder.build(
                state.messages, state.turn_summaries, state.chat_history_compaction
            )
            updates["turn_summaries"] = turn_context.turn_summaries
            updates["chat_history_compaction"] = turn_context.chat_history_compaction
            system = self._build_system_context(state)
            messages = system + turn_context.messages
        else:
            system = self._build_system_context(state)
            messages = system + [
                m.render() if isinstance(m, RenderableMessageMixin) else m for m in state.messages
            ]

        _raw = await self._get_llm_with_tools(state).ainvoke(messages, config)
        response = AgentMessage.from_langchain(_raw)
        updates["messages"] = [response]
        return updates


class TaskUpdateSyncNode(BaseNode):
    """Graph node that folds finished background tasks into context mid-turn.

    Sits on the ``tools -> agent`` edge: drains whatever background tasks
    completed since the last drain from *agent_task_manager* and turns each
    into a message via *task_message_from_record*, so a task that finishes
    mid-turn is woven into context before the next reasoning step rather
    than waiting for the turn to end.
    """

    def __init__(
        self,
        agent_task_manager: AgentTaskManagerBase,
        task_message_from_record: Callable[[AgentTaskRecord], TaskMessage],
    ) -> None:
        self._agent_task_manager = agent_task_manager
        self._task_message_from_record = task_message_from_record

    async def ainvoke(
        self, state: AgentState, config: Optional[RunnableConfig] = None
    ) -> dict[str, Any]:
        records = await self._agent_task_manager.gather_task_updates()
        if not records:
            return {}
        return {"messages": [self._task_message_from_record(record) for record in records]}
