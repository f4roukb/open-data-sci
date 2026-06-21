"""Unit tests for opendatasci.config.OpenDataSciConfig."""


from pathlib import Path

import pytest
from pydantic_core import ValidationError

from opendatasci.configs import DEFAULT_MODEL, DEFAULT_SECONDARY_MODEL, OpenDataSciConfig
from opendatasci.skills.local import _BUILTIN_SKILLS_DIRECTORY

# ---------------------------------------------------------------------------
# Basic construction and defaults
# ---------------------------------------------------------------------------


class TestAgentConfigDefaults:
    def test_default_provider_is_anthropic(self) -> None:
        assert OpenDataSciConfig().provider == "anthropic"

    def test_default_model_is_anthropic_provider_default(self) -> None:
        assert OpenDataSciConfig().model == DEFAULT_MODEL["anthropic"]

    def test_default_secondary_provider_is_anthropic(self) -> None:
        assert OpenDataSciConfig().secondary_provider == "anthropic"

    def test_default_secondary_model_is_anthropic_provider_default(self) -> None:
        assert OpenDataSciConfig().secondary_model == DEFAULT_SECONDARY_MODEL["anthropic"]

    def test_resolved_model_falls_back_to_provider_default(self) -> None:
        cfg = OpenDataSciConfig(provider="anthropic")
        assert cfg.model == DEFAULT_MODEL["anthropic"]

    def test_resolved_model_uses_explicit_value(self) -> None:
        cfg = OpenDataSciConfig(provider="anthropic", model="claude-opus-4-7")
        assert cfg.model == "claude-opus-4-7"


# ---------------------------------------------------------------------------
# secondary_provider resolution
# ---------------------------------------------------------------------------


class TestResolvedSecondaryModelProvider:
    def test_defaults_to_main_provider_when_not_set(self) -> None:
        cfg = OpenDataSciConfig(provider="anthropic")
        assert cfg.secondary_provider == "anthropic"

    def test_uses_explicit_secondary_provider(self) -> None:
        cfg = OpenDataSciConfig(provider="anthropic", secondary_provider="openai")
        assert cfg.secondary_provider == "openai"

    def test_resolved_secondary_model_uses_config_default(self) -> None:
        cfg = OpenDataSciConfig(provider="anthropic", secondary_provider="openai")
        assert cfg.secondary_model == DEFAULT_SECONDARY_MODEL["anthropic"]

    def test_resolved_secondary_model_uses_explicit_secondary_model(self) -> None:
        cfg = OpenDataSciConfig(
            provider="anthropic",
            secondary_provider="openai",
            secondary_model="gpt-4o-mini",
        )
        assert cfg.secondary_model == "gpt-4o-mini"

    def test_same_provider_both_models(self) -> None:
        cfg = OpenDataSciConfig(provider="openai", model="gpt-5.5", secondary_model="gpt-4o-mini")
        assert cfg.secondary_provider == "anthropic"
        assert cfg.secondary_model == "gpt-4o-mini"

    @pytest.mark.parametrize(
        "main_provider,secondary_provider",
        [
            ("anthropic", "openai"),
            ("openai", "anthropic"),
            ("anthropic", "gemini"),
            ("bedrock", "openai"),
        ],
    )
    def test_cross_provider_pairs(self, main_provider: str, secondary_provider: str) -> None:
        cfg = OpenDataSciConfig(provider=main_provider, secondary_provider=secondary_provider)  # type: ignore[arg-type]
        assert cfg.secondary_provider == secondary_provider
        assert cfg.secondary_model == DEFAULT_SECONDARY_MODEL["anthropic"]


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestAgentConfigValidation:
    def test_invalid_provider_raises(self) -> None:
        with pytest.raises(ValidationError):
            OpenDataSciConfig(provider="nonexistent")  # type: ignore[arg-type]

    def test_invalid_secondary_provider_raises(self) -> None:
        with pytest.raises(ValidationError):
            OpenDataSciConfig(secondary_provider="nonexistent")  # type: ignore[arg-type]

    def test_valid_providers_do_not_raise(self) -> None:
        for provider in DEFAULT_MODEL:
            OpenDataSciConfig(provider=provider)  # type: ignore[arg-type]

    def test_valid_secondary_providers_do_not_raise(self) -> None:
        for provider in DEFAULT_MODEL:
            OpenDataSciConfig(secondary_provider=provider)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# from_yaml
# ---------------------------------------------------------------------------


class TestAgentConfigFromYaml:
    def test_loads_provider_and_model(self, tmp_path: Path) -> None:
        f = tmp_path / "cfg.yaml"
        f.write_text("provider: openai\nmodel: gpt-4o\n")
        cfg = OpenDataSciConfig.from_yaml(f)
        assert cfg.provider == "openai"
        assert cfg.model == "gpt-4o"

    def test_loads_secondary_provider(self, tmp_path: Path) -> None:
        f = tmp_path / "cfg.yaml"
        f.write_text(
            "provider: anthropic\nsecondary_provider: openai\nsecondary_model: gpt-4o-mini\n"
        )
        cfg = OpenDataSciConfig.from_yaml(f)
        assert cfg.secondary_provider == "openai"
        assert cfg.secondary_model == "gpt-4o-mini"

    def test_loads_scalar_fields(self, tmp_path: Path) -> None:
        f = tmp_path / "cfg.yaml"
        f.write_text("temperature: 0.5\nthinking_budget: 2000\nname: TestBot\n")
        cfg = OpenDataSciConfig.from_yaml(f)
        assert cfg.temperature == 0.5
        assert cfg.thinking_budget == 2000
        assert cfg.name == "TestBot"

    def test_loads_list_fields(self, tmp_path: Path) -> None:
        f = tmp_path / "cfg.yaml"
        f.write_text("extra_web_domains:\n  - arxiv.org\n  - kaggle.com\n")
        cfg = OpenDataSciConfig.from_yaml(f)
        assert cfg.extra_web_domains == ["arxiv.org", "kaggle.com"]

    def test_empty_yaml_yields_defaults(self, tmp_path: Path) -> None:
        f = tmp_path / "cfg.yaml"
        f.write_text("")
        cfg = OpenDataSciConfig.from_yaml(f)
        assert cfg.provider == "anthropic"
        assert cfg.model == DEFAULT_MODEL["anthropic"]

    def test_accepts_path_object(self, tmp_path: Path) -> None:
        f = tmp_path / "cfg.yaml"
        f.write_text("provider: openai\n")
        cfg = OpenDataSciConfig.from_yaml(f)
        assert cfg.provider == "openai"

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        f = tmp_path / "cfg.yaml"
        f.write_text("provider: openai\n")
        cfg = OpenDataSciConfig.from_yaml(str(f))
        assert cfg.provider == "openai"

    def test_unknown_key_raises_value_error(self, tmp_path: Path) -> None:
        f = tmp_path / "cfg.yaml"
        f.write_text("provider: anthropic\nunknown_field: oops\n")
        with pytest.raises(ValueError, match="Unknown fields"):
            OpenDataSciConfig.from_yaml(f)

    def test_non_mapping_yaml_raises_value_error(self, tmp_path: Path) -> None:
        f = tmp_path / "cfg.yaml"
        f.write_text("- item1\n- item2\n")
        with pytest.raises(ValueError, match="must be a mapping"):
            OpenDataSciConfig.from_yaml(f)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            OpenDataSciConfig.from_yaml(tmp_path / "does_not_exist.yaml")

    def test_invalid_provider_in_yaml_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "cfg.yaml"
        f.write_text("provider: bogus\n")
        with pytest.raises(ValidationError):
            OpenDataSciConfig.from_yaml(f)

    def test_worker_timeout_none_preserved(self, tmp_path: Path) -> None:
        f = tmp_path / "cfg.yaml"
        f.write_text("worker_timeout_seconds: null\n")
        cfg = OpenDataSciConfig.from_yaml(f)
        assert cfg.worker_timeout_seconds is None


# ---------------------------------------------------------------------------
# Skills directory fields
# ---------------------------------------------------------------------------


class TestSkillsDirectoryConfig:
    def test_skills_directory_defaults_to_none(self) -> None:
        assert OpenDataSciConfig().skills_directory is None

    def test_BUILTIN_SKILLS_DIRECTORYectory_defaults_to_bundled_path(self) -> None:
        cfg = OpenDataSciConfig()
        assert cfg.builtin_skills_directory == _BUILTIN_SKILLS_DIRECTORY
        assert cfg.builtin_skills_directory.is_dir()

    def test_skills_directory_accepts_path(self, tmp_path: Path) -> None:
        cfg = OpenDataSciConfig(skills_directory=tmp_path)
        assert cfg.skills_directory == tmp_path

    def test_BUILTIN_SKILLS_DIRECTORYectory_accepts_custom_path(self, tmp_path: Path) -> None:
        cfg = OpenDataSciConfig(builtin_skills_directory=tmp_path)
        assert cfg.builtin_skills_directory == tmp_path

    def test_from_yaml_loads_skills_directory(self, tmp_path: Path) -> None:
        skills_dir = tmp_path / "custom_skills"
        skills_dir.mkdir()
        f = tmp_path / "cfg.yaml"
        f.write_text(f"skills_directory: {skills_dir.as_posix()}\n")
        cfg = OpenDataSciConfig.from_yaml(f)
        assert cfg.skills_directory == skills_dir

    def test_from_yaml_loads_BUILTIN_SKILLS_DIRECTORYectory(self, tmp_path: Path) -> None:
        custom_builtin = tmp_path / "builtin"
        custom_builtin.mkdir()
        f = tmp_path / "cfg.yaml"
        f.write_text(f"builtin_skills_directory: {custom_builtin.as_posix()}\n")
        cfg = OpenDataSciConfig.from_yaml(f)
        assert cfg.builtin_skills_directory == custom_builtin
