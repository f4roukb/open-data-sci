"""Unit tests for opendatasci._tui.commands — pure string-formatting logic."""

import pytest

from opendatasci._tui.commands import (
    SLASH_COMMANDS,
    SLASH_COMMAND_DESCRIPTIONS,
    _fmt_model,
    format_help_message,
    format_models_message,
    format_themes_message,
)


# ---------------------------------------------------------------------------
# _fmt_model
# ---------------------------------------------------------------------------


class TestFmtModel:
    """_fmt_model must turn provider/model pairs into human-readable labels."""

    def test_anthropic_claude_sonnet_formatted(self) -> None:
        result = _fmt_model("anthropic", "claude-sonnet-4-6")
        assert result == "Anthropic Claude Sonnet 4.6"

    def test_anthropic_claude_haiku_formatted(self) -> None:
        result = _fmt_model("anthropic", "claude-haiku-4-5-20251001")
        # Regex matches the first claude-[a-z]+-\d+-\d+ group: haiku, 4, 5
        assert result == "Anthropic Claude Haiku 4.5"

    def test_anthropic_claude_opus_formatted(self) -> None:
        result = _fmt_model("anthropic", "claude-opus-4-8")
        assert result == "Anthropic Claude Opus 4.8"

    def test_openai_non_claude_model_kept_as_is(self) -> None:
        result = _fmt_model("openai", "gpt-4o")
        assert "OpenAI" in result
        assert "gpt-4o" in result

    def test_bedrock_uses_display_name(self) -> None:
        result = _fmt_model("bedrock", "amazon-titan-v1")
        assert "AWS Bedrock" in result

    def test_google_gemini_uses_display_name(self) -> None:
        result = _fmt_model("gemini", "gemini-pro")
        assert "Google" in result

    def test_unknown_provider_falls_back_to_title_case(self) -> None:
        result = _fmt_model("my_vendor", "custom-model")
        # Provider("my_vendor") raises ValueError → title() applied
        assert "My_Vendor" in result
        assert "custom-model" in result

    def test_model_without_claude_pattern_returned_raw(self) -> None:
        # "claude-3-haiku-20240307" has no [a-z]+ after the first "-", so no regex match
        result = _fmt_model("anthropic", "claude-3-haiku-20240307")
        assert "Anthropic" in result
        assert "claude-3-haiku-20240307" in result

    def test_result_is_a_string(self) -> None:
        assert isinstance(_fmt_model("anthropic", "claude-sonnet-4-6"), str)

    def test_vertexai_uses_display_name(self) -> None:
        result = _fmt_model("vertexai", "gemini-1.5-pro")
        assert "Google Vertex AI" in result

    def test_azure_uses_display_name(self) -> None:
        result = _fmt_model("azure", "gpt-4o-mini")
        assert "Azure OpenAI" in result

    def test_ollama_uses_display_name(self) -> None:
        result = _fmt_model("ollama", "llama3")
        assert "Ollama" in result


# ---------------------------------------------------------------------------
# format_help_message
# ---------------------------------------------------------------------------


class TestFormatHelpMessage:
    """format_help_message must list all registered commands with descriptions."""

    def test_contains_available_commands_header(self) -> None:
        msg = format_help_message()
        assert "Available Commands" in msg

    def test_lists_all_slash_commands(self) -> None:
        msg = format_help_message()
        for cmd in SLASH_COMMANDS:
            assert cmd in msg, f"Command {cmd!r} missing from help message"

    def test_contains_at_file_tip(self) -> None:
        msg = format_help_message()
        assert "@" in msg

    def test_contains_slash_tip(self) -> None:
        msg = format_help_message()
        assert "/" in msg

    def test_is_markdown_formatted(self) -> None:
        msg = format_help_message()
        # Markdown header
        assert "##" in msg

    def test_exit_command_described(self) -> None:
        msg = format_help_message()
        assert "/exit" in msg

    def test_clear_command_described(self) -> None:
        msg = format_help_message()
        assert "/clear" in msg

    def test_help_command_described(self) -> None:
        msg = format_help_message()
        assert "/help" in msg


# ---------------------------------------------------------------------------
# format_models_message
# ---------------------------------------------------------------------------


class TestFormatModelsMessage:
    """format_models_message must show primary and secondary model information."""

    def test_contains_models_header(self) -> None:
        msg = format_models_message("anthropic", "claude-sonnet-4-6", "anthropic", "claude-haiku-4-5")
        assert "Model" in msg

    def test_primary_model_present(self) -> None:
        msg = format_models_message("anthropic", "claude-sonnet-4-6", "anthropic", "claude-haiku-4-5")
        assert "Sonnet" in msg

    def test_secondary_model_present(self) -> None:
        msg = format_models_message("anthropic", "claude-sonnet-4-6", "anthropic", "claude-haiku-4-5")
        assert "Haiku" in msg

    def test_primary_and_secondary_are_distinguishable(self) -> None:
        msg = format_models_message("anthropic", "claude-sonnet-4-6", "openai", "gpt-4o")
        # Both providers should appear somewhere
        assert "Anthropic" in msg
        assert "OpenAI" in msg

    def test_is_markdown_formatted(self) -> None:
        msg = format_models_message("anthropic", "claude-sonnet-4-6", "anthropic", "claude-haiku-4-5")
        assert "##" in msg

    def test_unknown_provider_does_not_raise(self) -> None:
        msg = format_models_message("custom", "model-1-0", "custom", "model-0-1")
        assert isinstance(msg, str)

    def test_primary_label_present(self) -> None:
        msg = format_models_message("anthropic", "claude-sonnet-4-6", "anthropic", "claude-haiku-4-5")
        assert "Primary" in msg

    def test_secondary_label_present(self) -> None:
        msg = format_models_message("anthropic", "claude-sonnet-4-6", "anthropic", "claude-haiku-4-5")
        assert "Secondary" in msg


# ---------------------------------------------------------------------------
# format_themes_message
# ---------------------------------------------------------------------------


class TestFormatThemesMessage:
    """format_themes_message must mark the active theme and list all options."""

    def test_active_theme_has_active_marker(self) -> None:
        themes = {"default": "Clean dark theme", "light": "Light theme"}
        msg = format_themes_message("default", themes)
        assert "*(active)*" in msg

    def test_only_active_theme_is_marked(self) -> None:
        themes = {"default": "A", "light": "B", "solarized": "C"}
        msg = format_themes_message("light", themes)
        assert msg.count("*(active)*") == 1

    def test_inactive_themes_not_marked(self) -> None:
        themes = {"default": "A", "light": "B"}
        msg = format_themes_message("default", themes)
        lines = msg.splitlines()
        light_line = next(l for l in lines if "light" in l)
        assert "*(active)*" not in light_line

    def test_all_theme_names_listed(self) -> None:
        themes = {"default": "A", "dracula": "B", "solarized": "C"}
        msg = format_themes_message("default", themes)
        for name in themes:
            assert name in msg

    def test_all_theme_descriptions_listed(self) -> None:
        themes = {"default": "Clean dark theme", "light": "Bright light palette"}
        msg = format_themes_message("default", themes)
        for desc in themes.values():
            assert desc in msg

    def test_contains_relaunch_tip(self) -> None:
        msg = format_themes_message("default", {"default": "A"})
        assert "--theme" in msg

    def test_has_markdown_header(self) -> None:
        msg = format_themes_message("default", {"default": "A"})
        assert "##" in msg

    def test_empty_themes_dict_returns_header_only(self) -> None:
        msg = format_themes_message("default", {})
        assert "##" in msg
        assert "*(active)*" not in msg


# ---------------------------------------------------------------------------
# SLASH_COMMANDS registry invariants
# ---------------------------------------------------------------------------


class TestSlashCommandsRegistry:
    """The registry itself must satisfy basic structural invariants."""

    def test_all_commands_start_with_slash(self) -> None:
        for cmd in SLASH_COMMANDS:
            assert cmd.startswith("/"), f"{cmd!r} does not start with /"

    def test_no_duplicate_commands(self) -> None:
        assert len(SLASH_COMMANDS) == len(set(SLASH_COMMANDS))

    def test_all_commands_have_descriptions(self) -> None:
        for cmd in SLASH_COMMANDS:
            assert cmd in SLASH_COMMAND_DESCRIPTIONS, f"{cmd!r} has no description"

    def test_descriptions_are_non_empty_strings(self) -> None:
        for cmd, desc in SLASH_COMMAND_DESCRIPTIONS.items():
            assert isinstance(desc, str) and desc.strip(), f"{cmd!r} has empty description"
