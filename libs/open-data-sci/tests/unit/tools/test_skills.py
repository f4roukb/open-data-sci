"""Unit tests for opendatasci.tools.skills."""


import json
from pathlib import Path

from langchain_core.messages import ToolMessage
from langgraph.types import Command

from opendatasci.agents.states import AgentState
from opendatasci.skills import LocalSkillStore
from opendatasci.skills.base import BaseSkillStore, Skill
from opendatasci.skills.local import _BUILTIN_SKILLS_DIRECTORY
from opendatasci.tools.skills import create_skill_tools

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CALL_ID = "test_call_id"


def _make_store(skills_dir: Path | None = None) -> BaseSkillStore:
    if skills_dir is not None:
        return LocalSkillStore([_BUILTIN_SKILLS_DIRECTORY, skills_dir])
    return LocalSkillStore()


def _invoke(tool, args: dict, *, state: AgentState | None = None) -> Command:
    """Invoke a tool in ToolCall format, injecting state manually."""
    call_args = {**args, "state": state if state is not None else AgentState()}
    return tool.invoke({"name": tool.name, "id": _CALL_ID, "args": call_args, "type": "tool_call"})


def _message_content(result: Command) -> str:
    msgs = result.update.get("messages", [])
    return msgs[0].content if msgs else ""


# ---------------------------------------------------------------------------
# LocalSkillStore via tools layer
# ---------------------------------------------------------------------------


class TestSkillCatalog:
    def test_builtin_skills_are_available(self) -> None:
        skills = LocalSkillStore().list()
        assert "data_science" in skills
        assert "machine_learning" in skills

    def test_builtin_count(self) -> None:
        assert len(LocalSkillStore().list()) == 6

    def test_user_defined_skill_is_merged(self, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        (skills_dir / "custom.json").write_text(
            json.dumps({"name": "custom", "prompt": "custom prompt"})
        )
        skills = LocalSkillStore([_BUILTIN_SKILLS_DIRECTORY, skills_dir]).list()
        assert "custom" in skills

    def test_user_defined_skill_overrides_builtin(self, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        (skills_dir / "data_science.json").write_text(
            json.dumps({"name": "data_science", "prompt": "override"})
        )
        skills = LocalSkillStore([_BUILTIN_SKILLS_DIRECTORY, skills_dir]).list()
        assert skills["data_science"].content == "override"


# ---------------------------------------------------------------------------
# create_skill_tools / get_skill_tool
# ---------------------------------------------------------------------------


class TestGetSkillTools:
    def test_returns_list_with_one_tool(self) -> None:
        tools = create_skill_tools(_make_store())
        assert len(tools) == 1
        assert tools[0].name == "load_skill"

    def test_create_skill_tools_returns_load_skill(self) -> None:
        assert create_skill_tools(_make_store())[0].name == "load_skill"


class TestLoadSkillTool:
    def test_already_loaded_skill_returns_message(self) -> None:
        tool = create_skill_tools(_make_store())[0]
        skill = Skill(name="data_science", content="x")
        result = _invoke(
            tool,
            {"skill": "data_science", "summary": "s", "communication": "c"},
            state=AgentState(active_skills=[skill]),
        )
        assert "already loaded" in _message_content(result)

    def test_unknown_skill_returns_error_with_available_list(self) -> None:
        result = _invoke(
            create_skill_tools(_make_store())[0],
            {"skill": "nonexistent", "summary": "s", "communication": "c"},
        )
        assert "Unknown skill" in _message_content(result)
        assert "data_science" in _message_content(result)

    def test_loading_known_skill_returns_confirmation(self) -> None:
        result = _invoke(
            create_skill_tools(_make_store())[0],
            {"skill": "data_science", "summary": "s", "communication": "c"},
        )
        assert "loaded" in _message_content(result).lower()

    def test_loading_skill_sets_active_skills_in_state(self) -> None:
        result = _invoke(
            create_skill_tools(_make_store())[0],
            {"skill": "machine_learning", "summary": "s", "communication": "c"},
        )
        assert isinstance(result, Command)
        skills = result.update.get("active_skills", [])
        assert len(skills) == 1
        assert skills[0].name == "machine_learning"

    def test_switching_skill_replaces_previous(self) -> None:
        tool = create_skill_tools(_make_store())[0]
        existing = Skill(name="data_science", content="x")
        result = _invoke(
            tool,
            {"skill": "machine_learning", "summary": "s", "communication": "c"},
            state=AgentState(active_skills=[existing]),
        )
        skills = result.update.get("active_skills", [])
        assert skills[0].name == "machine_learning"

    def test_command_includes_tool_message_with_correct_id(self) -> None:
        result = _invoke(
            create_skill_tools(_make_store())[0],
            {"skill": "data_science", "summary": "s", "communication": "c"},
        )
        msgs = result.update.get("messages", [])
        assert len(msgs) == 1
        assert isinstance(msgs[0], ToolMessage)
        assert msgs[0].tool_call_id == _CALL_ID

    def test_error_message_lists_all_available_skills(self) -> None:
        result = _invoke(
            create_skill_tools(_make_store())[0],
            {"skill": "bad_skill", "summary": "s", "communication": "c"},
        )
        content = _message_content(result)
        for name in ("data_science", "machine_learning", "quantitative_analysis"):
            assert name in content
