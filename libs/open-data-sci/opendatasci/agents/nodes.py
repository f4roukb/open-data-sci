from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, Optional

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig

from opendatasci.agents.chat_memory import ChatHistoryBuilder, render_messages_for_llm
from opendatasci.agents.states import AgentState
from opendatasci.models.factory import _RetryRunnable

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
                state.messages, state.turn_summaries
            )
            updates["turn_summaries"] = turn_context.turn_summaries
            system = self._build_system_context(state)
            messages = system + turn_context.messages
        else:
            system = self._build_system_context(state)
            messages = system + render_messages_for_llm(list(state.messages))

        response = await self._get_llm_with_tools(state).ainvoke(messages, config)
        updates["messages"] = [response]
        return updates
