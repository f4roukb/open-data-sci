"""Unit tests for opendatasci.skills (LocalSkillStore)."""


import json
from pathlib import Path

import pytest

from opendatasci.skills import Skill
from opendatasci.skills.local import _BUILTIN_SKILLS_DIRECTORY, LocalSkillStore, SKILL_LABELS


class TestBuiltinSkills:
    def test_skills_registry_has_all_expected_keys(self) -> None:
        expected = {
            "data_science",
            "competitive_data_science",
            "machine_learning",
            "deep_learning",
            "quantitative_analysis",
            "data_science_education",
        }
        assert set(LocalSkillStore().list().keys()) == expected

    def test_skill_labels_matches_builtin_skills_keys(self) -> None:
        builtin_keys = set(LocalSkillStore().list().keys())
        assert builtin_keys.issubset(set(SKILL_LABELS.keys()))

    def test_all_skills_have_non_empty_content(self) -> None:
        for name, skill in LocalSkillStore().list().items():
            assert skill.content.strip(), f"Skill '{name}' has empty content"

    def test_all_skills_are_skill_instances(self) -> None:
        for name, skill in LocalSkillStore().list().items():
            assert isinstance(skill, Skill), f"Skill '{name}' is not a Skill instance"

    def test_competitive_skill_content_is_distinct_from_base(self) -> None:
        skills = LocalSkillStore().list()
        assert skills["data_science"].content != skills["competitive_data_science"].content

    def test_skill_labels_are_non_empty_strings(self) -> None:
        for name, label in SKILL_LABELS.items():
            assert label, f"Label for '{name}' is empty"

    def test_contents_contain_meaningful_content(self) -> None:
        for name, skill in LocalSkillStore().list().items():
            assert len(skill.content) > 200, f"Skill '{name}' content is suspiciously short"

    def test_default_paths_is_builtin(self) -> None:
        loader = LocalSkillStore()
        assert loader._paths == [_BUILTIN_SKILLS_DIRECTORY]

    def test_explicit_none_paths_uses_builtin(self) -> None:
        loader = LocalSkillStore(None)
        assert loader._paths == [_BUILTIN_SKILLS_DIRECTORY]


class TestSkillDataclass:
    def test_create_skill(self) -> None:
        skill = Skill(name="my_skill", content="Do the thing.")
        assert skill.name == "my_skill"
        assert skill.content == "Do the thing."

    def test_skill_equality(self) -> None:
        assert Skill(name="s", content="x") == Skill(name="s", content="x")
        assert Skill(name="s", content="x") != Skill(name="s", content="y")
        assert Skill(name="a", content="x") != Skill(name="b", content="x")

    def test_all_builtin_skill_names_match_key(self) -> None:
        for key, skill in LocalSkillStore().list().items():
            assert skill.name == key


class TestLocalSkillStoreUserDefined:
    def test_empty_directory_returns_empty_dict(self, tmp_path: Path) -> None:
        result = LocalSkillStore([tmp_path]).load_user_defined()
        assert result == {}

    def test_nonexistent_directory_returns_empty_dict(self, tmp_path: Path) -> None:
        result = LocalSkillStore([tmp_path / "nonexistent"]).load_user_defined()
        assert result == {}

    def test_load_json_skill(self, tmp_path: Path) -> None:
        data = {"name": "my_skill", "prompt": "Do the thing.", "label": "My Skill"}
        (tmp_path / "my_skill.json").write_text(json.dumps(data))
        result = LocalSkillStore([tmp_path]).load_user_defined()
        assert "my_skill" in result
        assert result["my_skill"].content == "Do the thing."

    def test_skip_files_with_unsupported_extension(self, tmp_path: Path) -> None:
        (tmp_path / "notes.txt").write_text("whatever")
        (tmp_path / "data.csv").write_text("a,b,c")
        result = LocalSkillStore([tmp_path]).load_user_defined()
        assert result == {}

    def test_skip_invalid_json(self, tmp_path: Path) -> None:
        (tmp_path / "bad.json").write_text("not valid json {{{")
        result = LocalSkillStore([tmp_path], strict=False).load_user_defined()
        assert result == {}

    def test_skip_file_missing_name_key(self, tmp_path: Path) -> None:
        data = {"prompt": "Do something."}
        (tmp_path / "skill.json").write_text(json.dumps(data))
        result = LocalSkillStore([tmp_path], strict=False).load_user_defined()
        assert result == {}

    def test_skip_file_missing_prompt_key(self, tmp_path: Path) -> None:
        data = {"name": "skill1"}
        (tmp_path / "skill.json").write_text(json.dumps(data))
        result = LocalSkillStore([tmp_path], strict=False).load_user_defined()
        assert result == {}

    def test_skip_non_dict_json(self, tmp_path: Path) -> None:
        (tmp_path / "skill.json").write_text(json.dumps(["a", "b"]))
        result = LocalSkillStore([tmp_path], strict=False).load_user_defined()
        assert result == {}

    def test_skip_json_with_empty_name(self, tmp_path: Path) -> None:
        data = {"name": "", "prompt": "Do something."}
        (tmp_path / "skill.json").write_text(json.dumps(data))
        result = LocalSkillStore([tmp_path], strict=False).load_user_defined()
        assert result == {}

    def test_skip_json_with_empty_prompt(self, tmp_path: Path) -> None:
        data = {"name": "skill1", "prompt": ""}
        (tmp_path / "skill.json").write_text(json.dumps(data))
        result = LocalSkillStore([tmp_path], strict=False).load_user_defined()
        assert result == {}

    def test_strict_mode_raises_on_any_error(self, tmp_path: Path) -> None:
        (tmp_path / "bad.json").write_text("not valid json")
        with pytest.raises(ValueError):
            LocalSkillStore([tmp_path], strict=True).load_user_defined()

    def test_strict_mode_succeeds_with_valid_files(self, tmp_path: Path) -> None:
        data = {"name": "ok_skill", "prompt": "valid prompt"}
        (tmp_path / "ok.json").write_text(json.dumps(data))
        result = LocalSkillStore([tmp_path], strict=True).load_user_defined()
        assert "ok_skill" in result

    def test_multiple_skills_loaded(self, tmp_path: Path) -> None:
        for i in range(3):
            data = {"name": f"skill{i}", "prompt": f"prompt for skill {i}"}
            (tmp_path / f"skill{i}.json").write_text(json.dumps(data))
        result = LocalSkillStore([tmp_path]).load_user_defined()
        assert len(result) == 3

    def test_valid_and_invalid_mixed(self, tmp_path: Path) -> None:
        valid = {"name": "good_skill", "prompt": "good prompt"}
        (tmp_path / "good.json").write_text(json.dumps(valid))
        (tmp_path / "bad.json").write_text("not json")
        result = LocalSkillStore([tmp_path], strict=False).load_user_defined()
        assert "good_skill" in result
        assert len(result) == 1

    def test_strict_mode_raises_when_mixed_valid_and_invalid(self, tmp_path: Path) -> None:
        valid = {"name": "good_skill", "prompt": "prompt"}
        (tmp_path / "good.json").write_text(json.dumps(valid))
        (tmp_path / "bad.json").write_text("not json")
        with pytest.raises(ValueError):
            LocalSkillStore([tmp_path], strict=True).load_user_defined()

    def test_strict_mode_error_message_names_file(self, tmp_path: Path) -> None:
        (tmp_path / "broken.json").write_text("bad")
        with pytest.raises(ValueError, match="broken.json"):
            LocalSkillStore([tmp_path], strict=True).load_user_defined()

    def test_md_file_loaded_by_stem(self, tmp_path: Path) -> None:
        (tmp_path / "my_skill.md").write_text("# My Skill\nDo the thing.")
        result = LocalSkillStore([tmp_path]).load_user_defined()
        assert "my_skill" in result
        assert result["my_skill"].content == "# My Skill\nDo the thing."

    def test_user_defined_overrides_builtin(self, tmp_path: Path) -> None:
        data = {"name": "data_science", "prompt": "custom data science prompt"}
        (tmp_path / "data_science.json").write_text(json.dumps(data))
        skills = LocalSkillStore([_BUILTIN_SKILLS_DIRECTORY, tmp_path]).list()
        assert skills["data_science"].content == "custom data science prompt"

    def test_builtin_dir_excluded_from_load_user_defined(self, tmp_path: Path) -> None:
        data = {"name": "my_skill", "prompt": "user prompt"}
        (tmp_path / "my_skill.json").write_text(json.dumps(data))
        loader = LocalSkillStore([_BUILTIN_SKILLS_DIRECTORY, tmp_path])
        user = loader.load_user_defined()
        assert "my_skill" in user
        assert "data_science" not in user


class TestLocalSkillStoreMultiDir:
    def test_multiple_dirs_merged(self, tmp_path: Path) -> None:
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        (dir_a / "skill_a.json").write_text(json.dumps({"name": "skill_a", "prompt": "a"}))
        (dir_b / "skill_b.json").write_text(json.dumps({"name": "skill_b", "prompt": "b"}))
        result = LocalSkillStore([dir_a, dir_b]).list()
        assert "skill_a" in result
        assert "skill_b" in result

    def test_later_dir_overrides_earlier(self, tmp_path: Path) -> None:
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        (dir_a / "skill.json").write_text(json.dumps({"name": "skill", "prompt": "from_a"}))
        (dir_b / "skill.json").write_text(json.dumps({"name": "skill", "prompt": "from_b"}))
        result = LocalSkillStore([dir_a, dir_b]).list()
        assert result["skill"].content == "from_b"

    def test_empty_list_returns_empty_dict(self) -> None:
        result = LocalSkillStore([]).list()
        assert result == {}
