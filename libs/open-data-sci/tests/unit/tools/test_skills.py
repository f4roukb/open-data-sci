"""Unit tests for opendatasci.tools.skills."""


import json
from pathlib import Path

from langchain_core.messages import ToolMessage
from langgraph.types import Command

from opendatasci.agents.states import AgentState
from opendatasci.skills import LocalSkillStore
from opendatasci.skills.base import BaseSkillStore, Skill, SkillDomain
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


def _load_skill_tool(store: BaseSkillStore | None = None):
    return create_skill_tools(store if store is not None else _make_store())[0]


def _list_skills_tool(store: BaseSkillStore | None = None):
    return create_skill_tools(store if store is not None else _make_store())[1]


# ---------------------------------------------------------------------------
# LocalSkillStore via tools layer
# ---------------------------------------------------------------------------


class TestSkillCatalog:
    def test_builtin_domain_scoped_skills_are_available(self) -> None:
        skills = LocalSkillStore().list_skills()
        assert "competitive_data_science::reconnaissance" in skills
        assert "data_science::exploratory_analysis" in skills
        assert "machine_learning::problem_framing" in skills

    def test_builtin_domains_are_available(self) -> None:
        domains = LocalSkillStore().list_domains()
        assert "competitive_data_science" in domains
        assert "data_science" in domains
        assert "machine_learning" in domains

    def test_user_defined_skill_is_merged(self, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        (skills_dir / "custom.md").write_text("custom prompt")
        skills = LocalSkillStore([_BUILTIN_SKILLS_DIRECTORY, skills_dir]).list_skills()
        assert "custom" in skills

    def test_user_defined_skill_overrides_builtin(self, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills" / "data_science"
        skills_dir.mkdir(parents=True)
        (skills_dir / "exploratory_analysis.md").write_text("override")
        skills = LocalSkillStore([_BUILTIN_SKILLS_DIRECTORY, tmp_path / "skills"]).list_skills()
        assert skills["data_science::exploratory_analysis"].content == "override"


# ---------------------------------------------------------------------------
# create_skill_tools
# ---------------------------------------------------------------------------


class TestGetSkillTools:
    def test_returns_two_tools(self) -> None:
        tools = create_skill_tools(_make_store())
        assert len(tools) == 2

    def test_first_tool_is_load_skill(self) -> None:
        assert create_skill_tools(_make_store())[0].name == "load_skill"

    def test_second_tool_is_list_skills(self) -> None:
        assert create_skill_tools(_make_store())[1].name == "list_skills"


# ---------------------------------------------------------------------------
# list_skills
# ---------------------------------------------------------------------------


class TestListSkillsTool:
    def test_returns_json_with_domains(self) -> None:
        result = _list_skills_tool().invoke({"summary": "s", "communication": "c"})
        payload = json.loads(result)
        assert "competitive_data_science" in payload["domains"]
        assert "data_science" in payload["domains"]
        assert "machine_learning" in payload["domains"]

    def test_domain_scoped_skills_excluded_from_standalone_list(self) -> None:
        result = _list_skills_tool().invoke({"summary": "s", "communication": "c"})
        payload = json.loads(result)
        assert not any("::" in name for name in payload["standalone_skills"])
        assert "competitive_data_science::reconnaissance" not in payload["standalone_skills"]
        assert payload["standalone_skills"] == []


# ---------------------------------------------------------------------------
# load_skill
# ---------------------------------------------------------------------------


class TestLoadSkillTool:
    def test_no_args_returns_llm_friendly_error(self) -> None:
        result = _invoke(_load_skill_tool(), {"summary": "s", "communication": "c"})
        content = _message_content(result)
        assert "skill_domain_name" in content
        assert "skill_name" in content

    def test_already_loaded_skill_returns_message(self) -> None:
        skill = Skill(name="data_science::exploratory_analysis", content="x")
        result = _invoke(
            _load_skill_tool(),
            {"skill_name": "data_science::exploratory_analysis", "summary": "s", "communication": "c"},
            state=AgentState(active_skills=[skill]),
        )
        assert "already loaded" in _message_content(result)

    def test_unknown_skill_returns_error_with_available_list(self) -> None:
        result = _invoke(
            _load_skill_tool(),
            {"skill_name": "nonexistent", "summary": "s", "communication": "c"},
        )
        assert "Unknown skill" in _message_content(result)
        assert "data_science::exploratory_analysis" in _message_content(result)

    def test_loading_known_skill_returns_confirmation(self) -> None:
        result = _invoke(
            _load_skill_tool(),
            {"skill_name": "data_science::exploratory_analysis", "summary": "s", "communication": "c"},
        )
        assert "loaded" in _message_content(result).lower()

    def test_loading_skill_sets_active_skills_in_state(self) -> None:
        result = _invoke(
            _load_skill_tool(),
            {"skill_name": "machine_learning::problem_framing", "summary": "s", "communication": "c"},
        )
        assert isinstance(result, Command)
        skills = result.update.get("active_skills", [])
        assert len(skills) == 1
        assert skills[0].name == "machine_learning::problem_framing"

    def test_switching_skill_replaces_previous(self) -> None:
        existing = Skill(name="data_science::exploratory_analysis", content="x")
        result = _invoke(
            _load_skill_tool(),
            {"skill_name": "machine_learning::problem_framing", "summary": "s", "communication": "c"},
            state=AgentState(active_skills=[existing]),
        )
        skills = result.update.get("active_skills", [])
        assert skills[0].name == "machine_learning::problem_framing"

    def test_command_includes_tool_message_with_correct_id(self) -> None:
        result = _invoke(
            _load_skill_tool(),
            {"skill_name": "data_science::exploratory_analysis", "summary": "s", "communication": "c"},
        )
        msgs = result.update.get("messages", [])
        assert len(msgs) == 1
        assert isinstance(msgs[0], ToolMessage)
        assert msgs[0].tool_call_id == _CALL_ID

    def test_error_message_lists_all_available_skills(self) -> None:
        result = _invoke(
            _load_skill_tool(),
            {"skill_name": "bad_skill", "summary": "s", "communication": "c"},
        )
        content = _message_content(result)
        for name in (
            "data_science::exploratory_analysis",
            "machine_learning::problem_framing",
            "quantitative_analysis::problem_formulation",
        ):
            assert name in content

    def test_loading_domain_scoped_skill_by_qualified_name(self) -> None:
        result = _invoke(
            _load_skill_tool(),
            {
                "skill_name": "competitive_data_science::reconnaissance",
                "summary": "s",
                "communication": "c",
            },
        )
        skills = result.update.get("active_skills", [])
        assert skills[0].name == "competitive_data_science::reconnaissance"

    def test_loading_known_domain_returns_confirmation(self) -> None:
        result = _invoke(
            _load_skill_tool(),
            {"skill_domain_name": "competitive_data_science", "summary": "s", "communication": "c"},
        )
        domains = result.update.get("active_skill_domains")
        assert isinstance(domains, list)
        assert len(domains) == 1
        assert domains[0].name == "competitive_data_science"

    def test_unknown_domain_returns_error_with_available_list(self) -> None:
        result = _invoke(
            _load_skill_tool(),
            {"skill_domain_name": "nonexistent", "summary": "s", "communication": "c"},
        )
        content = _message_content(result)
        assert "Unknown skill domain" in content
        assert "competitive_data_science" in content

    def test_already_loaded_domain_returns_message(self) -> None:
        domain = SkillDomain(name="competitive_data_science", content="x")
        result = _invoke(
            _load_skill_tool(),
            {"skill_domain_name": "competitive_data_science", "summary": "s", "communication": "c"},
            state=AgentState(active_skill_domains=[domain]),
        )
        assert "already loaded" in _message_content(result)

    def test_loading_domain_and_skill_together(self) -> None:
        result = _invoke(
            _load_skill_tool(),
            {
                "skill_domain_name": "competitive_data_science",
                "skill_name": "competitive_data_science::reconnaissance",
                "summary": "s",
                "communication": "c",
            },
        )
        assert result.update["active_skill_domains"][0].name == "competitive_data_science"
        assert result.update["active_skills"][0].name == "competitive_data_science::reconnaissance"
