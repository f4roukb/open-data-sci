"""Unit tests for opendatasci.tools.tasks."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from opendatasci.configs import OpenDataSciConfig
from opendatasci.sandbox.base import BaseSandbox, BaseSandboxFactory
from opendatasci.skills.base import BaseSkillStore
from opendatasci.tasks.base import (
    BackgroundTaskProgressReport,
    BackgroundTaskProgressUpdate,
    BackgroundTaskRecord,
    BackgroundTaskStatus,
)
from opendatasci.tasks.local import BackgroundTaskManager
from opendatasci.tools.tasks import (
    CheckTaskTool,
    ListTasksTool,
    MonitorTaskTool,
    ReportProgressTool,
    StopTaskTool,
    TaskTool,
    create_task_management_tools,
    create_task_tools,
)
from opendatasci.workspace.base import BaseWorkspace

# ---------------------------------------------------------------------------
# Task management: check_task, list_tasks, stop_task
# ---------------------------------------------------------------------------


class TestCreateTaskManagementTools:
    def test_returns_check_list_and_cancel_tools(self) -> None:
        tools = create_task_management_tools(BackgroundTaskManager())
        names = {t.name for t in tools}
        assert names == {
            "check_task",
            "list_tasks",
            "stop_task",
            "monitor_task",
        }


class TestCheckTaskTool:
    @pytest.mark.asyncio
    async def test_unknown_task_id(self) -> None:
        tool = CheckTaskTool(background_task_manager=BackgroundTaskManager())
        unknown_id = uuid4()
        result = await tool.ainvoke({"task_id": str(unknown_id)})
        assert str(unknown_id) in result
        assert "no background task found" in result.lower()

    @pytest.mark.asyncio
    async def test_completed_task_reports_result(self) -> None:
        manager = BackgroundTaskManager()

        async def _work(task_id: object) -> str:
            return "the answer"

        task_id = await manager.submit_task(_work, summary="s")
        await asyncio.sleep(0)

        tool = CheckTaskTool(background_task_manager=manager)
        result = await tool.ainvoke({"task_id": str(task_id)})
        payload = json.loads(result)
        assert payload["status"] == "completed"
        assert payload["result"] == "the answer"

    @pytest.mark.asyncio
    async def test_failed_task_reports_error(self) -> None:
        manager = BackgroundTaskManager()

        async def _work(task_id: object) -> str:
            raise RuntimeError("boom")

        task_id = await manager.submit_task(_work, summary="s")
        await asyncio.sleep(0)

        tool = CheckTaskTool(background_task_manager=manager)
        result = await tool.ainvoke({"task_id": str(task_id)})
        payload = json.loads(result)
        assert payload["status"] == "failed"
        assert payload["error"] == "boom"

    @pytest.mark.asyncio
    async def test_includes_progress_reports(self) -> None:
        manager = BackgroundTaskManager()

        async def _work(task_id: object) -> str:
            await asyncio.sleep(10)
            return "never"

        task_id = await manager.submit_task(_work, summary="s")
        record = await manager.get_task(task_id)
        record.progress.append(
            BackgroundTaskProgressReport(
                progress_update=BackgroundTaskProgressUpdate(done="a", ongoing="b", blockers="c"),
                eta_seconds=15.0,
            ),
        )

        tool = CheckTaskTool(background_task_manager=manager)
        result = await tool.ainvoke({"task_id": str(task_id)})
        payload = json.loads(result)
        assert payload["progress"] == [
            {
                "done": "a",
                "ongoing": "b",
                "blockers": "c",
                "eta_seconds": 15.0,
                "reported_at": payload["progress"][0]["reported_at"],
            }
        ]

        await manager.cancel_task(task_id)

    @pytest.mark.asyncio
    async def test_includes_activity_log(self) -> None:
        manager = BackgroundTaskManager()

        async def _work(task_id: object) -> str:
            await asyncio.sleep(10)
            return "never"

        task_id = await manager.submit_task(_work, summary="s")
        await manager.push_activity(task_id, "tool: execute\nresult: 42")

        tool = CheckTaskTool(background_task_manager=manager)
        result = await tool.ainvoke({"task_id": str(task_id)})
        payload = json.loads(result)
        assert payload["activity"] == ["tool: execute\nresult: 42"]

        await manager.cancel_task(task_id)

    @pytest.mark.asyncio
    async def test_includes_monitoring_logs(self) -> None:
        manager = BackgroundTaskManager()

        async def _work(task_id: object) -> str:
            await asyncio.sleep(10)
            return "never"

        task_id = await manager.submit_task(_work, summary="s")
        (monitor_id,) = await manager.monitor_task(task_id, [r"error: \d+"])

        tool = CheckTaskTool(background_task_manager=manager)
        result = await tool.ainvoke({"task_id": str(task_id)})
        payload = json.loads(result)
        assert payload["monitors"] == (
            f"Monitoring logs:\n- Monitor({monitor_id}) Matches regex: error: \\d+\n"
        )

        await manager.cancel_task(task_id)

    @pytest.mark.asyncio
    async def test_monitoring_logs_present_with_no_active_monitors(self) -> None:
        manager = BackgroundTaskManager()

        async def _work(task_id: object) -> str:
            return "done"

        task_id = await manager.submit_task(_work, summary="s")
        await asyncio.sleep(0)

        tool = CheckTaskTool(background_task_manager=manager)
        result = await tool.ainvoke({"task_id": str(task_id)})
        payload = json.loads(result)
        assert payload["monitors"] == "Monitoring logs:\n"


class TestListTasksTool:
    @pytest.mark.asyncio
    async def test_defaults_to_running_only(self) -> None:
        manager = BackgroundTaskManager()

        async def _slow(task_id: object) -> str:
            await asyncio.sleep(10)
            return "never"

        async def _fast(task_id: object) -> str:
            return "done"

        running_id = await manager.submit_task(_slow, summary="running one")
        done_id = await manager.submit_task(_fast, summary="done one")
        await asyncio.sleep(0)

        tool = ListTasksTool(background_task_manager=manager)
        result = await tool.ainvoke({})
        assert str(running_id) in result
        assert str(done_id) not in result

        await manager.cancel_task(running_id)

    @pytest.mark.asyncio
    async def test_status_in_filters_explicitly(self) -> None:
        manager = BackgroundTaskManager()

        async def _work(task_id: object) -> str:
            return "done"

        task_id = await manager.submit_task(_work, summary="s")
        await asyncio.sleep(0)

        tool = ListTasksTool(background_task_manager=manager)
        result = await tool.ainvoke({"status_in": ["completed"]})
        assert str(task_id) in result

    @pytest.mark.asyncio
    async def test_no_matches_reports_empty(self) -> None:
        tool = ListTasksTool(background_task_manager=BackgroundTaskManager())
        result = await tool.ainvoke({})
        assert "no background tasks" in result.lower()

    @pytest.mark.asyncio
    async def test_show_monitors_false_by_default_omits_monitors(self) -> None:
        manager = BackgroundTaskManager()
        started = asyncio.Event()

        async def _work(task_id: object) -> str:
            started.set()
            await asyncio.sleep(10)
            return "never"

        task_id = await manager.submit_task(_work, summary="s")
        await asyncio.wait_for(started.wait(), timeout=1)
        await manager.monitor_task(task_id, ["pattern"])

        tool = ListTasksTool(background_task_manager=manager)
        result = await tool.ainvoke({})
        entries = json.loads(result)
        assert "monitors" not in entries[0]

        await manager.cancel_task(task_id)

    @pytest.mark.asyncio
    async def test_show_monitors_true_includes_monitoring_logs(self) -> None:
        manager = BackgroundTaskManager()
        started = asyncio.Event()

        async def _work(task_id: object) -> str:
            started.set()
            await asyncio.sleep(10)
            return "never"

        task_id = await manager.submit_task(_work, summary="s")
        await asyncio.wait_for(started.wait(), timeout=1)
        (monitor_id,) = await manager.monitor_task(task_id, ["pattern"])

        tool = ListTasksTool(background_task_manager=manager)
        result = await tool.ainvoke({"show_monitors": True})
        entries = json.loads(result)
        assert entries[0]["monitors"] == (
            f"Monitoring logs:\n- Monitor({monitor_id}) Matches regex: pattern\n"
        )

        await manager.cancel_task(task_id)


class TestStopTaskTool:
    @pytest.mark.asyncio
    async def test_unknown_task_id(self) -> None:
        tool = StopTaskTool(background_task_manager=BackgroundTaskManager())
        result = await tool.ainvoke({"task_id": str(uuid4())})
        assert "no background task found" in result.lower()

    @pytest.mark.asyncio
    async def test_stops_running_task(self) -> None:
        manager = BackgroundTaskManager()
        started = asyncio.Event()

        async def _work(task_id: object) -> str:
            started.set()
            await asyncio.sleep(10)
            return "never"

        task_id = await manager.submit_task(_work, summary="s")
        await asyncio.wait_for(started.wait(), timeout=1)

        tool = StopTaskTool(background_task_manager=manager)
        result = await tool.ainvoke({"task_id": str(task_id)})
        assert "stop requested" in result.lower()
        assert str(task_id) in result


class TestMonitorTaskTool:
    @pytest.mark.asyncio
    async def test_unknown_task_id(self) -> None:
        tool = MonitorTaskTool(background_task_manager=BackgroundTaskManager())
        unknown_id = uuid4()
        result = await tool.ainvoke({"task_id": str(unknown_id), "regex_patterns": ["a"]})
        assert "no background task found" in result.lower()

    @pytest.mark.asyncio
    async def test_registers_monitors_and_reports_their_ids(self) -> None:
        manager = BackgroundTaskManager()
        started = asyncio.Event()

        async def _work(task_id: object) -> str:
            started.set()
            await asyncio.sleep(10)
            return "never"

        task_id = await manager.submit_task(_work, summary="s")
        await asyncio.wait_for(started.wait(), timeout=1)

        tool = MonitorTaskTool(background_task_manager=manager)
        result = await tool.ainvoke({"task_id": str(task_id), "regex_patterns": ["a", "b"]})
        monitors = await manager.list_task_monitors(task_id)
        assert set(monitors.values()) == {"a", "b"}
        for monitor_id in monitors:
            assert str(monitor_id) in result

        await manager.cancel_task(task_id)

    @pytest.mark.asyncio
    async def test_invalid_pattern_returns_error_string_not_raised(self) -> None:
        manager = BackgroundTaskManager()
        started = asyncio.Event()

        async def _work(task_id: object) -> str:
            started.set()
            await asyncio.sleep(10)
            return "never"

        task_id = await manager.submit_task(_work, summary="s")
        await asyncio.wait_for(started.wait(), timeout=1)

        tool = MonitorTaskTool(background_task_manager=manager)
        result = await tool.ainvoke({"task_id": str(task_id), "regex_patterns": ["(unclosed"]})
        assert "invalid regex" in result.lower()
        assert await manager.list_task_monitors(task_id) == {}

        await manager.cancel_task(task_id)

    @pytest.mark.asyncio
    async def test_registering_the_same_pattern_twice_does_not_duplicate(self) -> None:
        manager = BackgroundTaskManager()
        started = asyncio.Event()

        async def _work(task_id: object) -> str:
            started.set()
            await asyncio.sleep(10)
            return "never"

        task_id = await manager.submit_task(_work, summary="s")
        await asyncio.wait_for(started.wait(), timeout=1)

        tool = MonitorTaskTool(background_task_manager=manager)
        await tool.ainvoke({"task_id": str(task_id), "regex_patterns": ["a"]})
        await tool.ainvoke({"task_id": str(task_id), "regex_patterns": ["a"]})

        assert len(await manager.list_task_monitors(task_id)) == 1

        await manager.cancel_task(task_id)



# ---------------------------------------------------------------------------
# Task creation: the `task` tool
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


class TestTaskDetails:
    def test_basic_construction(self) -> None:
        task = TaskTool.TaskDetails(subtask="Do something.", summary="Doing thing")
        assert task.subtask == "Do something."
        assert task.summary == "Doing thing"
        assert task.skill is None

    def test_skill_field_set(self) -> None:
        task = TaskTool.TaskDetails(
            subtask="Analyse data.", summary="Analyse", skill="data_science"
        )
        assert task.skill == "data_science"

    def test_skill_defaults_to_none(self) -> None:
        task = TaskTool.TaskDetails(subtask="x", summary="y")
        assert task.skill is None

    def test_missing_subtask_raises(self) -> None:
        with pytest.raises(Exception):
            TaskTool.TaskDetails(summary="only summary")  # type: ignore[call-arg]

    def test_missing_summary_raises(self) -> None:
        with pytest.raises(Exception):
            TaskTool.TaskDetails(subtask="only subtask")  # type: ignore[call-arg]


class TestCreateTaskToolsStructure:
    def test_returns_list_with_one_tool(self) -> None:
        tools = create_task_tools(
            _make_workspace(),
            datasci_config=OpenDataSciConfig(),
            sandbox_factory=_make_sandbox_factory(),
            background_task_manager=BackgroundTaskManager(),
            skill_store=_make_store(),
        )
        assert len(tools) == 1

    def test_tool_name_is_task(self) -> None:
        tools = create_task_tools(
            _make_workspace(),
            datasci_config=OpenDataSciConfig(),
            sandbox_factory=_make_sandbox_factory(),
            background_task_manager=BackgroundTaskManager(),
            skill_store=_make_store(),
        )
        assert tools[0].name == "task"


_AGENT_PATCH = "opendatasci.tools.tasks.WorkerAgent"


def _make_tool(**overrides) -> TaskTool:
    kwargs = {
        "workspace": _make_workspace(),
        "datasci_config": OpenDataSciConfig(),
        "sandbox_factory": _make_sandbox_factory(),
        "skill_store": _make_store(),
        "background_task_manager": BackgroundTaskManager(),
        **overrides,
    }
    return TaskTool(**kwargs)


class TestRunOne:
    @pytest.mark.asyncio
    async def test_returns_agent_output(self) -> None:
        tool = _make_tool()
        with patch(_AGENT_PATCH) as MockAgent:
            MockAgent.return_value.ainvoke = AsyncMock(return_value="direct output")
            result = await tool._arun_one(
                0,
                TaskTool.TaskDetails(subtask="Do X.", summary="X"),
                None,
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
                TaskTool.TaskDetails(subtask="Fail.", summary="F"),
                None,
                MagicMock(),
            )
        assert "boom" in result

    @pytest.mark.asyncio
    async def test_skill_resolved_from_store(self) -> None:
        store = _make_store()
        store.load = MagicMock(return_value="skill_obj")
        tool = _make_tool(skill_store=store)
        with patch(_AGENT_PATCH) as MockAgent:
            MockAgent.return_value.ainvoke = AsyncMock(return_value="ok")
            await tool._arun_one(
                0,
                TaskTool.TaskDetails(subtask="T.", summary="s", skill="data_science"),
                None,
                MagicMock(),
            )
        store.load.assert_called_once_with("data_science")

    @pytest.mark.asyncio
    async def test_unknown_skill_logs_warning_without_raising(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        store = _make_store()
        store.load = MagicMock(return_value=None)
        tool = _make_tool(skill_store=store)
        with patch(_AGENT_PATCH) as MockAgent:
            MockAgent.return_value.ainvoke = AsyncMock(return_value="ok")
            with caplog.at_level("WARNING"):
                result = await tool._arun_one(
                    0,
                    TaskTool.TaskDetails(subtask="T.", summary="s", skill="nonexistent"),
                    None,
                    MagicMock(),
                )
        assert result == "ok"
        assert "nonexistent" in caplog.text

    @pytest.mark.asyncio
    async def test_report_progress_tool_added_in_background_mode(self) -> None:
        tool = _make_tool()
        with patch(_AGENT_PATCH) as MockAgent:
            MockAgent.return_value.ainvoke = AsyncMock(return_value="ok")
            await tool._arun_one(
                0, TaskTool.TaskDetails(subtask="x", summary="y"), uuid4(), MagicMock()
            )
        _, kwargs = MockAgent.call_args
        assert "report_progress" in {t.name for t in kwargs["tools"]}

    @pytest.mark.asyncio
    async def test_report_progress_tool_absent_in_foreground_mode(self) -> None:
        tool = _make_tool()
        with patch(_AGENT_PATCH) as MockAgent:
            MockAgent.return_value.ainvoke = AsyncMock(return_value="ok")
            await tool._arun_one(
                0, TaskTool.TaskDetails(subtask="x", summary="y"), None, MagicMock()
            )
        _, kwargs = MockAgent.call_args
        assert "report_progress" not in {t.name for t in kwargs["tools"]}


class TestReportProgressTool:
    @pytest.mark.asyncio
    async def test_pushes_progress_to_manager(self) -> None:
        manager = BackgroundTaskManager()
        started = asyncio.Event()

        async def _work(task_id: object) -> str:
            started.set()
            await asyncio.sleep(10)
            return "never"

        task_id = await manager.submit_task(_work, summary="s")
        await asyncio.wait_for(started.wait(), timeout=1)

        tool = ReportProgressTool(task_id=task_id, background_task_manager=manager)
        result = await tool.ainvoke(
            {"done": "a", "ongoing": "b", "blockers": "c", "eta_seconds": 3.0}
        )

        assert "recorded" in result.lower()
        record = await manager.get_task(task_id)
        assert record is not None
        assert record.progress[-1].progress_update.done == "a"
        assert record.progress[-1].progress_update.ongoing == "b"
        assert record.progress[-1].progress_update.blockers == "c"
        assert record.progress[-1].eta_seconds == 3.0

        await manager.cancel_task(task_id)


class TestActivityLog:
    @pytest.mark.asyncio
    async def test_background_worker_tool_result_appends_activity(self) -> None:
        manager = BackgroundTaskManager()
        task_id = uuid4()
        await manager.upsert_record(
            BackgroundTaskRecord(task_id=task_id, summary="s", status=BackgroundTaskStatus.RUNNING)
        )
        tool = _make_tool(background_task_manager=manager)

        async def _fake_ainvoke(task, system_prompt, on_event=None, **kwargs):
            on_event("task_tool_result", "execute", {"success": True, "output": "42"})
            return "done"

        async def _noop_dispatch(*args, **kwargs) -> None:
            return None

        with (
            patch(_AGENT_PATCH) as MockAgent,
            patch("opendatasci.tools.tasks.adispatch_custom_event", _noop_dispatch),
        ):
            MockAgent.return_value.ainvoke = _fake_ainvoke
            await tool._arun_one(
                0, TaskTool.TaskDetails(subtask="x", summary="y"), task_id, MagicMock()
            )
            await _drain_emit_tasks()

        record = await manager.get_task(task_id)
        assert record is not None
        assert record.activity == ["tool: execute\nresult: 42"]

    @pytest.mark.asyncio
    async def test_foreground_worker_tool_result_does_not_raise(self) -> None:
        manager = BackgroundTaskManager()
        tool = _make_tool(background_task_manager=manager)

        async def _fake_ainvoke(task, system_prompt, on_event=None, **kwargs):
            on_event("task_tool_result", "execute", {"success": True, "output": "42"})
            return "done"

        async def _noop_dispatch(*args, **kwargs) -> None:
            return None

        with (
            patch(_AGENT_PATCH) as MockAgent,
            patch("opendatasci.tools.tasks.adispatch_custom_event", _noop_dispatch),
        ):
            MockAgent.return_value.ainvoke = _fake_ainvoke
            await tool._arun_one(
                0, TaskTool.TaskDetails(subtask="x", summary="y"), None, MagicMock()
            )
            await _drain_emit_tasks()

        assert await manager.list_tasks() == []


class TestTaskTool:
    def _get_tool(
        self,
        datasci_config: OpenDataSciConfig | None = None,
        store: MagicMock | None = None,
    ):
        tools = create_task_tools(
            _make_workspace(),
            datasci_config=datasci_config or OpenDataSciConfig(),
            sandbox_factory=_make_sandbox_factory(),
            skill_store=store or _make_store(),
            background_task_manager=BackgroundTaskManager(),
        )
        return tools[0]

    @pytest.mark.asyncio
    async def test_single_worker_result_returned(self) -> None:
        tool = self._get_tool()
        with patch(_AGENT_PATCH) as MockAgent:
            MockAgent.return_value.ainvoke = AsyncMock(return_value="worker output")
            result = await tool.ainvoke(
                {
                    "subtasks": [TaskTool.TaskDetails(subtask="Do X.", summary="Do X")],
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
                        TaskTool.TaskDetails(subtask="Task A.", summary="A"),
                        TaskTool.TaskDetails(subtask="Task B.", summary="B"),
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
                    "subtasks": [TaskTool.TaskDetails(subtask="Fail task.", summary="Fail")],
                    "summary": "s",
                    "communication": "spawning",
                }
            )
        assert "boom" in result

    @pytest.mark.asyncio
    async def test_task_done_event_dispatched_with_success(self) -> None:
        # Worker lifecycle signals are now dispatched into the caller's event
        # stream via adispatch_custom_event("task_event", ...) rather than via
        # side-channel queues. The "task_done" event must carry idx and success.
        recorded: list[dict] = []

        async def _record(name: str, payload: dict, **_: object) -> None:
            recorded.append(payload)

        tool = self._get_tool()
        with (
            patch(_AGENT_PATCH) as MockAgent,
            patch("opendatasci.tools.tasks.adispatch_custom_event", _record),
        ):
            MockAgent.return_value.ainvoke = AsyncMock(return_value="ok")
            await tool.ainvoke(
                {
                    "subtasks": [TaskTool.TaskDetails(subtask="Succeed.", summary="ok")],
                    "summary": "s",
                    "communication": "go",
                }
            )
            await _drain_emit_tasks()

        done = [p for p in recorded if p.get("event_type") == "task_done"]
        assert done
        assert done[0]["task_idx"] == 0
        assert done[0]["success"] is True

    @pytest.mark.asyncio
    async def test_task_started_event_dispatched(self) -> None:
        recorded: list[dict] = []

        async def _record(name: str, payload: dict, **_: object) -> None:
            recorded.append(payload)

        tool = self._get_tool()
        with (
            patch(_AGENT_PATCH) as MockAgent,
            patch("opendatasci.tools.tasks.adispatch_custom_event", _record),
        ):
            MockAgent.return_value.ainvoke = AsyncMock(return_value="done")
            await tool.ainvoke(
                {
                    "subtasks": [TaskTool.TaskDetails(subtask="Task.", summary="my task")],
                    "summary": "s",
                    "communication": "go",
                }
            )
            await _drain_emit_tasks()

        assert "task_started" in [p.get("event_type") for p in recorded]

    @pytest.mark.asyncio
    async def test_preloaded_skill_applied_to_worker_session(self) -> None:
        mock_store = MagicMock(spec=BaseSkillStore)
        mock_store.load = MagicMock(return_value=None)
        tools = create_task_tools(
            _make_workspace(),
            datasci_config=OpenDataSciConfig(),
            sandbox_factory=_make_sandbox_factory(),
            skill_store=mock_store,
            background_task_manager=BackgroundTaskManager(),
        )
        tool = tools[0]
        with patch(_AGENT_PATCH) as MockAgent:
            MockAgent.return_value.ainvoke = AsyncMock(return_value="done")
            await tool.ainvoke(
                {
                    "subtasks": [
                        TaskTool.TaskDetails(subtask="T.", summary="s", skill="data_science")
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
                        "subtasks": [TaskTool.TaskDetails(subtask="Slow.", summary="slow")],
                        "summary": "s",
                        "communication": "go",
                    }
                )


class TestRunMode:
    @pytest.mark.asyncio
    async def test_default_run_mode_returns_result_synchronously(self) -> None:
        tool = _make_tool()
        with patch(_AGENT_PATCH) as MockAgent:
            MockAgent.return_value.ainvoke = AsyncMock(return_value="worker output")
            result = await tool.ainvoke(
                {
                    "subtasks": [TaskTool.TaskDetails(subtask="Do X.", summary="X")],
                    "summary": "s",
                    "communication": "go",
                }
            )
        assert "worker output" in result

    @pytest.mark.asyncio
    async def test_background_mode_returns_immediately_with_task_id(self) -> None:
        tool = _make_tool()

        async def _slow_run(*args, **kwargs):
            await asyncio.sleep(10)
            return "never"

        with patch(_AGENT_PATCH) as MockAgent:
            MockAgent.return_value.ainvoke = _slow_run
            result = await asyncio.wait_for(
                tool.ainvoke(
                    {
                        "subtasks": [TaskTool.TaskDetails(subtask="Slow.", summary="slow")],
                        "summary": "s",
                        "communication": "go",
                        "run_mode": "background",
                    }
                ),
                timeout=1,
            )

        assert "scheduled" in result.lower()
        records = await tool.background_task_manager.list_tasks()
        assert len(records) == 1
        for record in records:
            await tool.background_task_manager.cancel_task(record.task_id)
        await asyncio.gather(
            *(t for t in tool.background_task_manager._tasks.values()), return_exceptions=True
        )

    @pytest.mark.asyncio
    async def test_background_mode_one_task_id_per_subtask(self) -> None:
        tool = _make_tool()

        async def _slow_run(*args, **kwargs):
            await asyncio.sleep(10)
            return "never"

        with patch(_AGENT_PATCH) as MockAgent:
            MockAgent.return_value.ainvoke = _slow_run
            result = await asyncio.wait_for(
                tool.ainvoke(
                    {
                        "subtasks": [
                            TaskTool.TaskDetails(subtask="Slow A.", summary="A"),
                            TaskTool.TaskDetails(subtask="Slow B.", summary="B"),
                        ],
                        "summary": "s",
                        "communication": "go",
                        "run_mode": "background",
                    }
                ),
                timeout=1,
            )

        assert "scheduled 2 background task" in result.lower()
        records = await tool.background_task_manager.list_tasks()
        assert len(records) == 2
        assert {r.summary for r in records} == {"A", "B"}
        assert len({r.task_id for r in records}) == 2

        for record in records:
            await tool.background_task_manager.cancel_task(record.task_id)
        await asyncio.gather(
            *(t for t in tool.background_task_manager._tasks.values()), return_exceptions=True
        )

    @pytest.mark.asyncio
    async def test_background_mode_runs_worker_in_background(self) -> None:
        tool = _make_tool()
        ran = asyncio.Event()

        async def _run(*args, **kwargs):
            ran.set()
            return "done"

        with patch(_AGENT_PATCH) as MockAgent:
            MockAgent.return_value.ainvoke = _run
            await tool.ainvoke(
                {
                    "subtasks": [TaskTool.TaskDetails(subtask="T.", summary="s")],
                    "summary": "s",
                    "communication": "go",
                    "run_mode": "background",
                }
            )
            await asyncio.wait_for(ran.wait(), timeout=1)

    @pytest.mark.asyncio
    async def test_background_mode_task_failure_is_logged_not_raised(self) -> None:
        tool = _make_tool()
        with patch(_AGENT_PATCH) as MockAgent:
            MockAgent.return_value.ainvoke = AsyncMock(side_effect=RuntimeError("boom"))
            result = await tool.ainvoke(
                {
                    "subtasks": [TaskTool.TaskDetails(subtask="Fail.", summary="s")],
                    "summary": "s",
                    "communication": "go",
                    "run_mode": "background",
                }
            )
            assert "scheduled" in result.lower()
            await _drain_emit_tasks()
