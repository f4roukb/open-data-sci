"""Unit tests for opendatasci.tools.workers."""


import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from opendatasci.tools.workers import WorkerTask, _run_one, create_worker_tools


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_workspace() -> MagicMock:
    wb = MagicMock()
    # _run_one does Path(workspace.get_reference()), so a real path-like is needed.
    wb.get_reference.return_value = "/tmp/workspace"
    return wb


async def _drain_emit_tasks() -> None:
    """Yield control so fire-and-forget emit tasks (scheduled via create_task)
    get a chance to run before assertions."""
    for _ in range(5):
        await asyncio.sleep(0)


def _make_sandbox_factory() -> MagicMock:
    """Return a mock sandbox factory whose context manager yields a mock sandbox."""
    factory = MagicMock()
    sandbox = MagicMock()
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=sandbox)
    cm.__aexit__ = AsyncMock(return_value=False)
    factory.create.return_value = cm
    return factory


def _make_store() -> MagicMock:
    store = MagicMock()
    store.load = MagicMock(return_value=None)
    return store


# ---------------------------------------------------------------------------
# WorkerTask model
# ---------------------------------------------------------------------------


class TestWorkerTask:
    def test_basic_construction(self) -> None:
        task = WorkerTask(subtask="Do something.", summary="Doing thing")
        assert task.subtask == "Do something."
        assert task.summary == "Doing thing"
        assert task.skill is None

    def test_skill_field_set(self) -> None:
        task = WorkerTask(subtask="Analyse data.", summary="Analyse", skill="data_science")
        assert task.skill == "data_science"

    def test_skill_defaults_to_none(self) -> None:
        task = WorkerTask(subtask="x", summary="y")
        assert task.skill is None

    def test_missing_subtask_raises(self) -> None:
        with pytest.raises(Exception):
            WorkerTask(summary="only summary")  # type: ignore[call-arg]

    def test_missing_summary_raises(self) -> None:
        with pytest.raises(Exception):
            WorkerTask(subtask="only subtask")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# create_worker_tools – structure
# ---------------------------------------------------------------------------


class TestGetWorkerToolsStructure:
    def test_returns_list_with_one_tool(self) -> None:
        tools = create_worker_tools(
            _make_workspace(), None, datasci_config=None, sandbox_factory=_make_sandbox_factory()
        )
        assert len(tools) == 1

    def test_tool_name_is_spawn_workers(self) -> None:
        tools = create_worker_tools(
            _make_workspace(), None, datasci_config=None, sandbox_factory=_make_sandbox_factory()
        )
        assert tools[0].name == "spawn_workers"


# ---------------------------------------------------------------------------
# _run_one – direct tests (possible now that it is module-level)
# ---------------------------------------------------------------------------

# ParallelWorkerAgent is imported locally inside _run_one to break the
# tools → agents → tools circular dependency, so it must be patched at its
# definition site, not at opendatasci.tools.workers.
_AGENT_PATCH = "opendatasci.agents.agents.ParallelWorkerAgent"


class TestRunOne:
    def _kwargs(self, **overrides):
        return {
            "sandbox_factory": _make_sandbox_factory(),
            "workspace": _make_workspace(),
            "store": _make_store(),
            "datasci_config": None,
            **overrides,
        }

    @pytest.mark.asyncio
    async def test_returns_agent_output(self) -> None:
        with patch(_AGENT_PATCH) as MockAgent:
            MockAgent.return_value.ainvoke = AsyncMock(return_value="direct output")
            config = MagicMock()
            result = await _run_one(
                0,
                WorkerTask(subtask="Do X.", summary="X"),
                config,
                **self._kwargs(),
            )
        assert result == "direct output"

    @pytest.mark.asyncio
    async def test_runtime_error_returned_as_string(self) -> None:
        with patch(_AGENT_PATCH) as MockAgent:
            MockAgent.return_value.ainvoke = AsyncMock(side_effect=RuntimeError("boom"))
            result = await _run_one(
                0,
                WorkerTask(subtask="Fail.", summary="F"),
                MagicMock(),
                **self._kwargs(),
            )
        assert "boom" in result

    @pytest.mark.asyncio
    async def test_skill_resolved_from_store(self) -> None:
        store = _make_store()
        store.load = MagicMock(return_value="skill_obj")
        with patch(_AGENT_PATCH) as MockAgent:
            MockAgent.return_value.ainvoke = AsyncMock(return_value="ok")
            await _run_one(
                0,
                WorkerTask(subtask="T.", summary="s", skill="data_science"),
                MagicMock(),
                **self._kwargs(store=store),
            )
        store.load.assert_called_once_with("data_science")


# ---------------------------------------------------------------------------
# create_worker_tools – spawn_workers behaviour
# ---------------------------------------------------------------------------


class TestSpawnWorkersTool:
    def _get_tool(
        self,
        datasci_config: MagicMock | None = None,
        store: MagicMock | None = None,
    ):
        tools = create_worker_tools(
            _make_workspace(),
            None,
            datasci_config=datasci_config,
            sandbox_factory=_make_sandbox_factory(),
            store=store,
        )
        return tools[0]

    @pytest.mark.asyncio
    async def test_single_worker_result_returned(self) -> None:
        tool = self._get_tool()
        with patch(_AGENT_PATCH) as MockAgent:
            MockAgent.return_value.ainvoke = AsyncMock(return_value="worker output")
            result = await tool.ainvoke(
                {
                    "subtasks": [WorkerTask(subtask="Do X.", summary="Do X")],
                    "communication": "spawning",
                }
            )
        assert "Do X." in result
        assert "worker output" in result

    @pytest.mark.asyncio
    async def test_multiple_workers_all_results_included(self) -> None:
        tool = self._get_tool()
        call_count = 0

        async def _run(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return f"output_{call_count}"

        with patch(_AGENT_PATCH) as MockAgent:
            MockAgent.return_value.ainvoke = _run
            result = await tool.ainvoke(
                {
                    "subtasks": [
                        WorkerTask(subtask="Task A.", summary="A"),
                        WorkerTask(subtask="Task B.", summary="B"),
                    ],
                    "communication": "spawning",
                }
            )
        assert "Task A." in result
        assert "Task B." in result

    @pytest.mark.asyncio
    async def test_worker_exception_reported_in_output(self) -> None:
        # RuntimeError from agent.ainvoke is caught inside _run_one and returned as
        # its string message; other exceptions propagate and get the "Error: worker
        # failed" prefix from spawn_workers.  Both paths include the message text.
        tool = self._get_tool()
        with patch(_AGENT_PATCH) as MockAgent:
            MockAgent.return_value.ainvoke = AsyncMock(side_effect=RuntimeError("boom"))
            result = await tool.ainvoke(
                {
                    "subtasks": [WorkerTask(subtask="Fail task.", summary="Fail")],
                    "communication": "spawning",
                }
            )
        assert "boom" in result

    @pytest.mark.asyncio
    async def test_worker_done_event_dispatched_with_success(self) -> None:
        # Worker lifecycle signals are now dispatched into the caller's event
        # stream via adispatch_custom_event("worker_event", ...) rather than via
        # side-channel queues. The "worker_done" event must carry idx and success.
        recorded: list[dict] = []

        async def _record(name: str, payload: dict, **_: object) -> None:
            recorded.append(payload)

        tool = self._get_tool()
        with (
            patch(_AGENT_PATCH) as MockAgent,
            patch("opendatasci.tools.workers.adispatch_custom_event", _record),
        ):
            MockAgent.return_value.ainvoke = AsyncMock(return_value="ok")
            await tool.ainvoke(
                {
                    "subtasks": [WorkerTask(subtask="Succeed.", summary="ok")],
                    "communication": "go",
                }
            )
            await _drain_emit_tasks()

        done = [p for p in recorded if p.get("event_type") == "worker_done"]
        assert done
        assert done[0]["worker_idx"] == 0
        assert done[0]["success"] is True

    @pytest.mark.asyncio
    async def test_worker_started_event_dispatched(self) -> None:
        recorded: list[dict] = []

        async def _record(name: str, payload: dict, **_: object) -> None:
            recorded.append(payload)

        tool = self._get_tool()
        with (
            patch(_AGENT_PATCH) as MockAgent,
            patch("opendatasci.tools.workers.adispatch_custom_event", _record),
        ):
            MockAgent.return_value.ainvoke = AsyncMock(return_value="done")
            await tool.ainvoke(
                {
                    "subtasks": [WorkerTask(subtask="Task.", summary="my task")],
                    "communication": "go",
                }
            )
            await _drain_emit_tasks()

        assert "worker_started" in [p.get("event_type") for p in recorded]

    @pytest.mark.asyncio
    async def test_preloaded_skill_applied_to_worker_session(self) -> None:
        mock_store = MagicMock()
        mock_store.load = MagicMock(return_value=None)
        tools = create_worker_tools(
            _make_workspace(),
            None,
            datasci_config=None,
            sandbox_factory=_make_sandbox_factory(),
            store=mock_store,
        )
        tool = tools[0]
        with patch(_AGENT_PATCH) as MockAgent:
            MockAgent.return_value.ainvoke = AsyncMock(return_value="done")
            await tool.ainvoke(
                {
                    "subtasks": [WorkerTask(subtask="T.", summary="s", skill="data_science")],
                    "communication": "go",
                }
            )
        mock_store.load.assert_called_once_with("data_science")

    @pytest.mark.asyncio
    async def test_timeout_uses_agent_config_value(self) -> None:
        config = MagicMock()
        config.worker_timeout_seconds = 0.01
        tool = self._get_tool(datasci_config=config)
        with patch(_AGENT_PATCH) as MockAgent:
            async def _slow_run(*args, **kwargs):
                await asyncio.sleep(10)
                return "never"

            MockAgent.return_value.ainvoke = _slow_run
            with pytest.raises(asyncio.TimeoutError):
                await tool.ainvoke(
                    {
                        "subtasks": [WorkerTask(subtask="Slow.", summary="slow")],
                        "communication": "go",
                    }
                )
