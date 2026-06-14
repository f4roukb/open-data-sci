"""Unit tests for opendatasci.agents.graph."""


from unittest.mock import MagicMock

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph

from opendatasci.agents.graphs import AgentGraphFactory


def _make_builder(**kwargs) -> AgentGraphFactory:
    _default_llm = MagicMock()
    defaults = {
        "get_llm_with_tools": lambda state: _default_llm,
        "tools": [],
        "build_system_context": lambda state, memory_text: [],
    }
    defaults.update(kwargs)
    return AgentGraphFactory(**defaults)


class TestAgentGraphFactory:
    def test_build_returns_compiled_state_graph(self) -> None:
        graph = _make_builder().build()
        assert isinstance(graph, CompiledStateGraph)

    def test_builds_with_empty_tool_list(self) -> None:
        graph = _make_builder(tools=[]).build()
        assert graph is not None

    def test_builds_with_no_checkpointer(self) -> None:
        graph = _make_builder(checkpointer=None).build()
        assert graph is not None

    def test_builds_with_checkpointer(self) -> None:
        graph = _make_builder(checkpointer=MemorySaver()).build()
        assert graph is not None

    def test_graph_has_agent_node(self) -> None:
        graph = _make_builder().build()
        assert "agent" in graph.nodes

    def test_graph_has_tools_node(self) -> None:
        graph = _make_builder().build()
        assert "tools" in graph.nodes

    def test_build_system_context_callable_accepted(self) -> None:
        called: list = []

        def build_system_context(state, memory_text) -> list:
            called.append((state, memory_text))
            return []

        graph = _make_builder(build_system_context=build_system_context).build()
        assert graph is not None
