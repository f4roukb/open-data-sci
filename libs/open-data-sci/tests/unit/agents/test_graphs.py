"""Unit tests for opendatasci.agents.graph."""


from unittest.mock import MagicMock

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph

from opendatasci.agents.graphs import AgentGraphFactory


def _make_builder(**kwargs) -> AgentGraphFactory:
    _default_llm = MagicMock()
    defaults = {
        "get_llm_with_tools": lambda state: _default_llm,
        "tools": [],
        "build_system_context": lambda state: [],
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

        def build_system_context(state) -> list:
            called.append(state)
            return []

        graph = _make_builder(build_system_context=build_system_context).build()
        assert graph is not None

    def test_no_sync_task_updates_node_when_not_supplied(self) -> None:
        graph = _make_builder().build()
        assert "sync_task_updates" not in graph.nodes

    def test_sync_task_updates_node_added_when_supplied(self) -> None:
        async def sync_task_updates(state, config=None) -> dict:
            return {}

        graph = _make_builder(sync_task_updates=sync_task_updates).build()
        assert "sync_task_updates" in graph.nodes

    def test_sync_task_updates_sits_between_tools_and_agent(self) -> None:
        async def sync_task_updates(state, config=None) -> dict:
            return {}

        graph = _make_builder(sync_task_updates=sync_task_updates).build()
        graph_repr = graph.get_graph()
        edges = {(e.source, e.target) for e in graph_repr.edges}
        assert ("tools", "sync_task_updates") in edges
        assert ("sync_task_updates", "agent") in edges
        assert ("tools", "agent") not in edges
