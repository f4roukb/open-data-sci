from typing import TYPE_CHECKING, Any, Callable

from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from opendatasci._utils.langchain_utils import is_final_ai_message
from opendatasci.agents.nodes import AgentNode, BuildSystemContext
from opendatasci.agents.states import AgentState
from opendatasci.models.factory import _RetryRunnable

if TYPE_CHECKING:
    from opendatasci.agents.chat_memory import ChatHistoryBuilder


def _route_after_llm_call(state: AgentState) -> str:
    return "end" if is_final_ai_message(state.messages[-1]) else "tools"


class AgentGraphFactory:
    """Builds the execution graph for the main agent."""

    def __init__(
        self,
        *,
        get_llm_with_tools: Callable[[AgentState], _RetryRunnable],
        tools: list[BaseTool],
        build_system_context: BuildSystemContext,
        chat_history_builder: "ChatHistoryBuilder | None" = None,
        checkpointer: "BaseCheckpointSaver[Any] | None" = None,
    ) -> None:
        self._get_llm_with_tools = get_llm_with_tools
        self._tools = tools
        self._build_system_context = build_system_context
        self._chat_history_builder = chat_history_builder
        self._checkpointer = checkpointer

    def build(self) -> CompiledStateGraph:
        """Compile and return the graph, ready to run."""
        agent_node = AgentNode(
            get_llm_with_tools=self._get_llm_with_tools,
            build_system_context=self._build_system_context,
            chat_history_builder=self._chat_history_builder,
        )

        graph = StateGraph(AgentState)
        graph.add_node("agent", agent_node.to_async_callable())
        graph.add_node("tools", ToolNode(self._tools))
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", _route_after_llm_call, {"tools": "tools", "end": END})
        graph.add_edge("tools", "agent")
        return graph.compile(checkpointer=self._checkpointer)


class WorkerGraphFactory:
    """Builds the execution graph for one-shot worker agents."""

    def __init__(
        self,
        *,
        llm_with_tools: _RetryRunnable,
        tools: list[BaseTool],
        build_system_context: BuildSystemContext,
    ) -> None:
        self._llm_with_tools = llm_with_tools
        self._tools = tools
        self._build_system_context = build_system_context

    def build(self) -> CompiledStateGraph:
        """Compile and return the worker graph, ready to run."""
        agent_node = AgentNode(
            get_llm_with_tools=lambda state: self._llm_with_tools,
            build_system_context=self._build_system_context,
        )

        graph = StateGraph(AgentState)
        graph.add_node("agent", agent_node.to_async_callable())
        graph.add_node("tools", ToolNode(self._tools))
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", _route_after_llm_call, {"tools": "tools", "end": END})
        graph.add_edge("tools", "agent")
        return graph.compile()
