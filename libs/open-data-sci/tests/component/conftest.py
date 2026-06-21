"""Component-test conftest: stubs native modules and provides shared fixtures.

Component tests exercise the TUI service layer with a *real* ``Agent``
wired together, but with external boundaries (LLM API, sandbox) replaced by
lightweight stubs.  The agent is wrapped in the production
``OpenDataSciTuiService`` exactly as the controller wires it at boot.
"""


# ---------------------------------------------------------------------------
# Stub unavailable native modules BEFORE any opendatasci imports
# ---------------------------------------------------------------------------
import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock


def _make_sandbox_runtime_stub() -> ModuleType:
    mod = ModuleType("sandbox_runtime")
    mod.__path__ = []  # mark as a package so `sandbox_runtime.utils.platform` resolves

    config_cls = MagicMock(name="SandboxRuntimeConfig")
    config_cls.return_value = MagicMock()

    manager_cls = MagicMock(name="SandboxManager")
    manager_cls.initialize = AsyncMock()
    manager_cls.reset = AsyncMock()
    manager_cls.wrap_with_sandbox = AsyncMock(return_value=("", "", 0))
    manager_cls.check_dependencies = MagicMock(return_value=True)
    manager_cls.is_supported_platform = MagicMock(return_value=True)

    mod.SandboxRuntimeConfig = config_cls  # type: ignore[attr-defined]
    mod.SandboxManager = manager_cls  # type: ignore[attr-defined]
    return mod


def _make_sandbox_runtime_utils_platform_stub() -> ModuleType:
    mod = ModuleType("sandbox_runtime.utils.platform")
    mod.get_platform = MagicMock(return_value="linux")  # type: ignore[attr-defined]
    return mod


if "sandbox_runtime" not in sys.modules:
    sys.modules["sandbox_runtime"] = _make_sandbox_runtime_stub()
    utils_mod = ModuleType("sandbox_runtime.utils")
    utils_mod.__path__ = []
    sys.modules["sandbox_runtime.utils"] = utils_mod
    sys.modules["sandbox_runtime.utils.platform"] = _make_sandbox_runtime_utils_platform_stub()

# ---------------------------------------------------------------------------
# Standard imports (after stub is in place)
# ---------------------------------------------------------------------------

from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Iterable
from unittest.mock import patch

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from opendatasci.agents.agents import Agent
from opendatasci.configs import OpenDataSciConfig
from opendatasci.context.local import LocalContextStore
from opendatasci.sandbox.base import BaseSandbox, BaseSandboxFactory, SandboxExecResult
from opendatasci.skills.local import LocalSkillStore
from opendatasci.tools import create_agent_tools
from opendatasci.workspace.local import LocalWorkspace
from opendatasci._tui.service import OpenDataSciTuiService

# ---------------------------------------------------------------------------
# Workspace fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def data_file(tmp_path):
    """A single CSV file, used as a single-file workspace source."""
    f = tmp_path / "sales.csv"
    f.write_text("product,revenue\nA,100\nB,200\n")
    return f


@pytest.fixture
def data_dir(tmp_path):
    """A directory with two CSV files, simulating a multi-file workspace."""
    (tmp_path / "sales.csv").write_text("product,revenue\nA,100\n")
    (tmp_path / "costs.csv").write_text("category,cost\nX,50\n")
    return tmp_path


# ---------------------------------------------------------------------------
# Sandbox stub
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_sandbox():
    """Minimal Sandbox stub that satisfies the agent interface without OS sandbox binaries."""
    sb = MagicMock(spec=BaseSandbox)
    sb.execute = AsyncMock(
        return_value=SandboxExecResult(
            success=True,
            output=None,
            stdout="",
            error=None,
            code="",
        )
    )
    sb.execute_cli = AsyncMock(
        return_value=SandboxExecResult(
            success=True,
            output=None,
            stdout="",
            error=None,
            code="",
        )
    )
    sb.close = AsyncMock()
    sb.reset = MagicMock()
    sb.get_history = MagicMock(return_value=[])
    return sb


class _StubSandboxFactory(BaseSandboxFactory):
    """Yields a pre-built sandbox stub from its async-context ``create``."""

    def __init__(self, sandbox: BaseSandbox) -> None:
        self._sandbox = sandbox

    def create(self, workspace_path: Path | None = None):
        sandbox = self._sandbox

        @asynccontextmanager
        async def _cm() -> AsyncIterator[BaseSandbox]:
            try:
                yield sandbox
            finally:
                await sandbox.close()

        return _cm()


# ---------------------------------------------------------------------------
# LLM stub
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm():
    """Fake LLM whose bind_tools() returns itself and whose ainvoke() returns a plain AIMessage."""
    llm = MagicMock()
    llm.bind_tools = MagicMock(return_value=llm)
    llm.ainvoke = AsyncMock(return_value=AIMessage(content="Analysis complete."))
    return llm


# ---------------------------------------------------------------------------
# Scripted real-graph chat model
# ---------------------------------------------------------------------------


class _ScriptedChatModel(GenericFakeChatModel):
    """``GenericFakeChatModel`` extended for OpenDataSci compatibility.

    * ``bind_tools`` is a no-op that returns ``self`` so ``Agent.__init__``
      can call ``llm.bind_tools(tools)`` without ``NotImplementedError``.
    * ``_stream`` is overridden so that scripted messages carrying ``tool_calls``
      but empty content still yield exactly one ``ChatGenerationChunk``. The
      base implementation only chunks ``content`` and ``additional_kwargs``,
      so a tool-call-only message would yield zero chunks and crash
      ``astream_events`` with "No generations found in stream."
    """

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):  # type: ignore[override]
        return self

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[override]
        import json

        from langchain_core.messages import AIMessageChunk
        from langchain_core.outputs import ChatGenerationChunk

        chat_result = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        message = chat_result.generations[0].message
        tool_call_chunks = [
            {
                "name": tc["name"],
                "args": json.dumps(tc["args"]),
                "id": tc.get("id"),
                "index": i,
            }
            for i, tc in enumerate(getattr(message, "tool_calls", []) or [])
        ]
        chunk = ChatGenerationChunk(
            message=AIMessageChunk(
                content=message.content,
                id=message.id,
                tool_call_chunks=tool_call_chunks,
            )
        )
        if run_manager and isinstance(message.content, str) and message.content:
            run_manager.on_llm_new_token(message.content, chunk=chunk)
        yield chunk


# ---------------------------------------------------------------------------
# Config fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def datasci_config() -> OpenDataSciConfig:
    return OpenDataSciConfig(provider="anthropic", model="claude-sonnet-4-6")


# ---------------------------------------------------------------------------
# Service builder
# ---------------------------------------------------------------------------


async def _build_entered_service(
    llm: Any,
    sandbox: BaseSandbox,
    path: str,
    config: OpenDataSciConfig,
) -> tuple[OpenDataSciTuiService, Agent]:
    """Build a real ``Agent`` driven by *llm* and wrap it in the TUI service.

    The agent's external boundaries are mocked: ``create_model`` returns *llm*,
    ``create_secondary_model`` returns ``None`` (no summarizer), ``with_retry``
    is identity, and ``tools.coding.create_model`` returns a structured-output
    capable stub for the ``verify_python_code`` tool.  Tools are built explicitly
    (with a sandbox factory and a ``save_plan`` callback) so worker spawning and
    plan-mode persistence work end-to-end.
    """
    workspace = LocalWorkspace(path)
    workspace_path = Path(workspace.get_reference())
    factory = _StubSandboxFactory(sandbox)
    context_store = LocalContextStore(workspace_path)
    skill_store = LocalSkillStore()
    session_id = "testsess"

    coding_llm = MagicMock()
    coding_llm.with_structured_output = MagicMock(return_value=AsyncMock())

    with patch("opendatasci.tools.coding.create_model", return_value=coding_llm):
        tools = create_agent_tools(
            workspace,
            sandbox,
            context_store,
            store=skill_store,
            datasci_config=config,
            sandbox_factory=factory,
            save_plan=lambda plan: context_store.save_plan(session_id, plan),
        )

    with (
        patch("opendatasci.agents.agents.create_model", return_value=llm),
        patch("opendatasci.agents.agents.create_secondary_model", return_value=None),
        patch("opendatasci.agents.agents.with_retry", side_effect=lambda x: x),
        patch("opendatasci.tools.coding.create_model", return_value=coding_llm),
    ):
        agent = Agent(
            workspace=workspace,
            session_id=session_id,
            context_store=context_store,
            skill_store=skill_store,
            sandbox_factory=factory,
            config=config,
            tools=tools,
        )
        await agent.__aenter__()

    service = OpenDataSciTuiService(
        agent=agent,
        sandbox=sandbox,
        workspace_path=workspace_path,
    )
    return service, agent


@pytest.fixture
async def loaded_opendatasci_service(mock_sandbox, mock_llm, datasci_config, data_file):
    """A TUI service backed by a real agent driven by the deterministic mock LLM."""
    service, agent = await _build_entered_service(
        mock_llm, mock_sandbox, str(data_file), datasci_config
    )
    try:
        yield service
    finally:
        await agent.__aexit__(None, None, None)


@pytest.fixture
async def make_scripted_service(mock_sandbox, datasci_config):
    """Factory: build a TUI service driven by a scripted real chat model.

    Tracks created agents and tears them down at the end of the test.
    """
    agents: list[Agent] = []

    async def _build(scripted_messages: Iterable[AIMessage], path: str) -> OpenDataSciTuiService:
        llm = _ScriptedChatModel(messages=iter(list(scripted_messages)))
        service, agent = await _build_entered_service(
            llm, mock_sandbox, path, datasci_config
        )
        agents.append(agent)
        return service

    yield _build

    for agent in agents:
        await agent.__aexit__(None, None, None)


@pytest.fixture
async def loaded_scripted_service(make_scripted_service, data_file):
    """Factory returning a *loaded* TUI service driven by scripted AIMessages."""

    async def _load(scripted_messages: Iterable[AIMessage]) -> OpenDataSciTuiService:
        return await make_scripted_service(scripted_messages, str(data_file))

    return _load
