"""Unit tests for opendatasci.agents.agents_factory."""


from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestCreateAgent:
    async def test_create_agent_constructs_components_from_path(self) -> None:
        """`create_agent` builds workspace/sandbox_factory/context/stores from a
        path and injects them into an Agent usable as an async context manager."""
        from opendatasci.agents.agents_factory import create_agent

        workspace_stub = MagicMock()
        workspace_stub.get_reference.return_value = "/tmp/fake_workspace"
        sandbox_stub = MagicMock()
        sandbox_stub.get_history = MagicMock(return_value=[])
        context_store_stub = MagicMock()
        context_store_stub.root = Path("/tmp/fake_workspace")
        skill_store_stub = MagicMock()

        @asynccontextmanager
        async def _fake_create(*_args, **_kwargs):
            yield sandbox_stub

        factory_stub = MagicMock()
        factory_stub.create = _fake_create

        with (
            patch("opendatasci.agents.agents_factory.LocalWorkspace", return_value=workspace_stub),
            patch("opendatasci.agents.agents_factory.SRTSandboxFactory", return_value=factory_stub),
            patch("opendatasci.agents.agents_factory.LocalContextStore", return_value=context_store_stub),
            patch("opendatasci.agents.agents_factory.LocalSkillStore", return_value=skill_store_stub),
            patch("opendatasci.agents.agents.with_retry", side_effect=lambda x: x),
            patch("opendatasci.agents.agents.create_agent_tools", return_value=[]),
            patch("opendatasci.agents.agents.AgentGraphFactory"),
            patch("opendatasci.agents.agents.create_model") as create_primary_mock,
            patch("opendatasci.agents.agents.create_secondary_model") as create_secondary_mock,
        ):
            llm_stub = MagicMock()
            llm_stub.bind_tools.side_effect = lambda _tools: MagicMock()
            create_primary_mock.return_value = llm_stub
            create_secondary_mock.return_value = llm_stub

            agent = create_agent("/some/data.csv")
            async with agent:
                assert agent._workspace is workspace_stub
                assert agent._sandbox is sandbox_stub
                assert agent._context_store is context_store_stub
                assert agent._skill_store is skill_store_stub
