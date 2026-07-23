"""Unit tests for opendatasci.tools.workers."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from opendatasci.configs import OpenDataSciConfig
from opendatasci.sandbox.base import BaseSandbox, BaseSandboxFactory
from opendatasci.skills.base import BaseSkillStore
from opendatasci.tasks.local import LocalTaskManager
from opendatasci.tools.workers import SpawnWorkersTool, create_worker_tools
from opendatasci.workspace.base import BaseWorkspace

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_workspace() -> MagicMock:
    wb = MagicMock(spec=BaseWorkspace)
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
    factory = MagicMock(spec=BaseSandboxFactory)
    sandbox = MagicMock(spec=BaseSandbox)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=sandbox)
    cm.__aexit__ = AsyncMock(return_value=False)
    factory.create.return_value = cm
    return factory


def _make_store() -> MagicMock:
    store = MagicMock(spec=BaseSkillStore)
    store.load = MagicMock(return_value=None)
    return store


# ---------------------------------------------------------------------------
# TaskDetails model
# ---------------------------------------------------------------------------


class TestTaskDetails:
    def test_basic_construction(self) -> None:
        task = SpawnWorkersTool.TaskDetails(subtask="Do something.", summary="Doing thing")
        assert task.subtask == "Do something."
        assert task.summary == "Doing thing"
        assert task.skill is None

    def test_skill_field_set(self) -> None:
        task = SpawnWorkersTool.TaskDetails(
            subtask="Analyse data.", summary="Analyse", skill="data_science"
        )
        assert task.skill == "data_science"

    def test_skill_defaults_to_none(self) -> None:
        task = SpawnWorkersTool.TaskDetails(subtask="x", summary="y")
        assert task.skill is None

    def test_missing_subtask_raises(self) -> None:
        with pytest.raises(Exception):
            SpawnWorkersTool.TaskDetails(summary="only summary")  # type: ignore[call-arg]

    def test_missing_summary_raises(self) -> None:
        with pytest.raises(Exception):
            SpawnWorkersTool.TaskDetails(subtask="only subtask")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# create_worker_tools – structure
# ---------------------------------------------------------------------------


class TestGetWorkerToolsStructure:
    def test_returns_list_with_one_tool(self) -> None:
        tools = create_worker_tools(
            _make_workspace(),
            None,
            datasci_config=None,
            sandbox_factory=_make_sandbox_factory(),
            task_manager=LocalTaskManager(),
        )
        assert len(tools) == 1

    def test_tool_name_is_task(self) -> None:
        tools = create_worker_tools(
            _make_workspace(),
            None,
            datasci_config=None,
            sandbox_factory=_make_sandbox_factory(),
            task_manager=LocalTaskManager(),
        )
        assert tools[0].name == "task"


# ---------------------------------------------------------------------------
# _arun_one – direct tests via SpawnWorkersTool instance
# ---------------------------------------------------------------------------

# ConcurrentWorkerAgent is imported locally inside _arun_one to break the
# tools → agents → tools circular dependency, so it must be patched at its
# definition site, not at opendatasci.tools.workers.
_AGENT_PATCH = "opendatasci.agents.agents.ConcurrentWorkerAgent"


def _make_tool(**overrides) -> SpawnWorkersTool:
    kwargs = {
        "workspace": _make_workspace(),
        "datasci_config": None,
        "sandbox_factory": _make_sandbox_factory(),
        "store": _make_store(),
        "task_manager": LocalTaskManager(),
        **overrides,
    }
    return SpawnWorkersTool(**kwargs)


class TestRunOne:
    @pytest.mark.asyncio
    async def test_returns_agent_output(self) -> None:
        tool = _make_tool()
        with patch(_AGENT_PATCH) as MockAgent:
            MockAgent.return_value.ainvoke = AsyncMock(return_value="direct output")
            result = await tool._arun_one(
                0,
                SpawnWorkersTool.TaskDetails(subtask="Do X.", summary="X"),
                MagicMock(),
            )
        assert result == "direct output"

    @pytest.mark.asyncio
    async def test_runtime_error_returned_as_string(self) -> None:
        tool = _make_tool()
        with patch(_AGENT_PATCH) as MockAgent:
            MockAgent.return_value.ainvoke = AsyncMock(side_effect=RuntimeError("boom"))
            result = await tool._arun_one(
                0,
                SpawnWorkersTool.TaskDetails(subtask="Fail.", summary="F"),
                MagicMock(),
            )
        assert "boom" in result

    @pytest.mark.asyncio
    async def test_skill_resolved_from_store(self) -> None:
        store = _make_store()
        store.load = MagicMock(return_value="skill_obj")
        tool = _make_tool(store=store)
        with patch(_AGENT_PATCH) as MockAgent:
            MockAgent.return_value.ainvoke = AsyncMock(return_value="ok")
            await tool._arun_one(
                0,
                SpawnWorkersTool.TaskDetails(subtask="T.", summary="s", skill="data_science"),
                MagicMock(),
            )
        store.load.assert_called_once_with("data_science")


# ---------------------------------------------------------------------------
# create_worker_tools – task behaviour
# ---------------------------------------------------------------------------


class TestSpawnWorkersTool:
    def _get_tool(
        self,
        datasci_config: OpenDataSciConfig | None = None,
        store: MagicMock | None = None,
    ):
        tools = create_worker_tools(
            _make_workspace(),
            None,
            datasci_config=datasci_config or OpenDataSciConfig(),
            sandbox_factory=_make_sandbox_factory(),
            store=store,
            task_manager=LocalTaskManager(),
        )
        return tools[0]

    @pytest.mark.asyncio
    async def test_single_worker_result_returned(self) -> None:
        tool = self._get_tool()
        with patch(_AGENT_PATCH) as MockAgent:
            MockAgent.return_value.ainvoke = AsyncMock(return_value="worker output")
            result = await tool.ainvoke(
                {
                    "subtasks": [SpawnWorkersTool.TaskDetails(subtask="Do X.", summary="Do X")],
                    "summary": "s",
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
                        SpawnWorkersTool.TaskDetails(subtask="Task A.", summary="A"),
                        SpawnWorkersTool.TaskDetails(subtask="Task B.", summary="B"),
                    ],
                    "summary": "s",
                    "communication": "spawning",
                }
            )
        assert "Task A." in result
        assert "Task B." in result

    @pytest.mark.asyncio
    async def test_worker_exception_reported_in_output(self) -> None:
        # RuntimeError from agent.ainvoke is caught inside _run_one and returned as
        # its string message; other exceptions propagate and get the "Error: worker
        # failed" prefix from task.  Both paths include the message text.
        tool = self._get_tool()
        with patch(_AGENT_PATCH) as MockAgent:
            MockAgent.return_value.ainvoke = AsyncMock(side_effect=RuntimeError("boom"))
            result = await tool.ainvoke(
                {
                    "subtasks": [
                        SpawnWorkersTool.TaskDetails(subtask="Fail task.", summary="Fail")
                    ],
                    "summary": "s",
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
                    "subtasks": [SpawnWorkersTool.TaskDetails(subtask="Succeed.", summary="ok")],
                    "summary": "s",
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
                    "subtasks": [SpawnWorkersTool.TaskDetails(subtask="Task.", summary="my task")],
                    "summary": "s",
                    "communication": "go",
                }
            )
            await _drain_emit_tasks()

        assert "worker_started" in [p.get("event_type") for p in recorded]

    @pytest.mark.asyncio
    async def test_preloaded_skill_applied_to_worker_session(self) -> None:
        mock_store = MagicMock(spec=BaseSkillStore)
        mock_store.load = MagicMock(return_value=None)
        tools = create_worker_tools(
            _make_workspace(),
            None,
            datasci_config=OpenDataSciConfig(),
            sandbox_factory=_make_sandbox_factory(),
            store=mock_store,
            task_manager=LocalTaskManager(),
        )
        tool = tools[0]
        with patch(_AGENT_PATCH) as MockAgent:
            MockAgent.return_value.ainvoke = AsyncMock(return_value="done")
            await tool.ainvoke(
                {
                    "subtasks": [
                        SpawnWorkersTool.TaskDetails(
                            subtask="T.", summary="s", skill="data_science"
                        )
                    ],
                    "summary": "s",
                    "communication": "go",
                }
            )
        mock_store.load.assert_called_once_with("data_science")

    @pytest.mark.asyncio
    async def test_timeout_uses_agent_config_value(self) -> None:
        config = OpenDataSciConfig(worker_timeout_seconds=0.01)
        tool = self._get_tool(datasci_config=config)
        with patch(_AGENT_PATCH) as MockAgent:

            async def _slow_run(*args, **kwargs):
                await asyncio.sleep(10)
                return "never"

            MockAgent.return_value.ainvoke = _slow_run
            with pytest.raises(asyncio.TimeoutError):
                await tool.ainvoke(
                    {
                        "subtasks": [SpawnWorkersTool.TaskDetails(subtask="Slow.", summary="slow")],
                        "summary": "s",
                        "communication": "go",
                    }
                )


# ---------------------------------------------------------------------------
# run_mode – parallel vs. concurrent worker execution
# ---------------------------------------------------------------------------


class TestRunMode:
    def test_default_run_mode_is_concurrent(self) -> None:
        tool = _make_tool()
        assert tool.run_mode == "concurrent"

    def test_run_mode_forwarded_by_create_worker_tools(self) -> None:
        tools = create_worker_tools(
            _make_workspace(),
            None,
            datasci_config=None,
            sandbox_factory=_make_sandbox_factory(),
            task_manager=LocalTaskManager(),
            run_mode="parallel",
        )
        assert tools[0].run_mode == "parallel"

    @pytest.mark.asyncio
    async def test_parallel_mode_runs_all_subtasks_and_collects_results(self) -> None:
        tool = _make_tool(run_mode="parallel")

        async def _run(*args, **kwargs):
            return "output"

        with patch(_AGENT_PATCH) as MockAgent:
            MockAgent.return_value.ainvoke = _run
            result = await tool.ainvoke(
                {
                    "subtasks": [
                        SpawnWorkersTool.TaskDetails(subtask="Task A.", summary="A"),
                        SpawnWorkersTool.TaskDetails(subtask="Task B.", summary="B"),
                    ],
                    "summary": "s",
                    "communication": "spawning",
                }
            )
        assert "Task A." in result
        assert "Task B." in result
        assert result.count("output") == 2

    @pytest.mark.asyncio
    async def test_parallel_mode_uses_parallel_worker(self) -> None:
        tool = _make_tool(run_mode="parallel")
        with (
            patch(_AGENT_PATCH) as MockAgent,
            patch("opendatasci.tools.workers.ParallelWorker") as MockParallelWorker,
            patch("opendatasci.tools.workers.ConcurrentWorker") as MockConcurrentWorker,
        ):
            MockAgent.return_value.ainvoke = AsyncMock(return_value="ok")
            MockParallelWorker.return_value.run = AsyncMock(return_value=["ok"])
            await tool.ainvoke(
                {
                    "subtasks": [SpawnWorkersTool.TaskDetails(subtask="T.", summary="s")],
                    "summary": "s",
                    "communication": "go",
                }
            )
        MockParallelWorker.assert_called_once()
        MockConcurrentWorker.assert_not_called()

    @pytest.mark.asyncio
    async def test_worker_exception_reported_in_parallel_mode(self) -> None:
        tool = _make_tool(run_mode="parallel")
        with patch(_AGENT_PATCH) as MockAgent:
            MockAgent.return_value.ainvoke = AsyncMock(side_effect=RuntimeError("boom"))
            result = await tool.ainvoke(
                {
                    "subtasks": [
                        SpawnWorkersTool.TaskDetails(subtask="Fail task.", summary="Fail")
                    ],
                    "summary": "s",
                    "communication": "spawning",
                }
            )
        assert "boom" in result

    @pytest.mark.asyncio
    async def test_worker_events_dispatched_in_parallel_mode(self) -> None:
        recorded: list[dict] = []

        async def _record(name: str, payload: dict, **_: object) -> None:
            recorded.append(payload)

        tool = _make_tool(run_mode="parallel")
        with (
            patch(_AGENT_PATCH) as MockAgent,
            patch("opendatasci.tools.workers.adispatch_custom_event", _record),
        ):
            MockAgent.return_value.ainvoke = AsyncMock(return_value="ok")
            await tool.ainvoke(
                {
                    "subtasks": [
                        SpawnWorkersTool.TaskDetails(subtask="Succeed A.", summary="A"),
                        SpawnWorkersTool.TaskDetails(subtask="Succeed B.", summary="B"),
                    ],
                    "summary": "s",
                    "communication": "go",
                }
            )
            await _drain_emit_tasks()

        done = [p for p in recorded if p.get("event_type") == "worker_done"]
        assert {d["worker_idx"] for d in done} == {0, 1}
        assert all(d["success"] for d in done)


# ---------------------------------------------------------------------------
# synch_mode – sync (blocking) vs. async (background-scheduled) execution
# ---------------------------------------------------------------------------


class TestSynchMode:
    @pytest.mark.asyncio
    async def test_default_synch_mode_returns_result_synchronously(self) -> None:
        tool = _make_tool()
        with patch(_AGENT_PATCH) as MockAgent:
            MockAgent.return_value.ainvoke = AsyncMock(return_value="worker output")
            result = await tool.ainvoke(
                {
                    "subtasks": [SpawnWorkersTool.TaskDetails(subtask="Do X.", summary="X")],
                    "summary": "s",
                    "communication": "go",
                }
            )
        assert "worker output" in result

    @pytest.mark.asyncio
    async def test_async_mode_returns_immediately_with_task_id(self) -> None:
        tool = _make_tool()

        async def _slow_run(*args, **kwargs):
            await asyncio.sleep(10)
            return "never"

        with patch(_AGENT_PATCH) as MockAgent:
            MockAgent.return_value.ainvoke = _slow_run
            result = await asyncio.wait_for(
                tool.ainvoke(
                    {
                        "subtasks": [SpawnWorkersTool.TaskDetails(subtask="Slow.", summary="slow")],
                        "summary": "s",
                        "communication": "go",
                        "synch_mode": "async",
                    }
                ),
                timeout=1,
            )

        assert "scheduled" in result.lower()
        records = await tool.task_manager.list()
        assert len(records) == 1
        for record in records:
            await tool.task_manager.cancel(record.task_id)
        await asyncio.gather(
            *(t for t in tool.task_manager._asyncio_tasks.values()), return_exceptions=True
        )

    @pytest.mark.asyncio
    async def test_async_mode_runs_worker_in_background(self) -> None:
        tool = _make_tool()
        ran = asyncio.Event()

        async def _run(*args, **kwargs):
            ran.set()
            return "done"

        with patch(_AGENT_PATCH) as MockAgent:
            MockAgent.return_value.ainvoke = _run
            await tool.ainvoke(
                {
                    "subtasks": [SpawnWorkersTool.TaskDetails(subtask="T.", summary="s")],
                    "summary": "s",
                    "communication": "go",
                    "synch_mode": "async",
                }
            )
            await asyncio.wait_for(ran.wait(), timeout=1)

    @pytest.mark.asyncio
    async def test_async_mode_background_task_failure_is_logged_not_raised(self) -> None:
        tool = _make_tool()
        with patch(_AGENT_PATCH) as MockAgent:
            MockAgent.return_value.ainvoke = AsyncMock(side_effect=RuntimeError("boom"))
            result = await tool.ainvoke(
                {
                    "subtasks": [SpawnWorkersTool.TaskDetails(subtask="Fail.", summary="s")],
                    "summary": "s",
                    "communication": "go",
                    "synch_mode": "async",
                }
            )
            assert "scheduled" in result.lower()
            await _drain_emit_tasks()
