"""Unit tests for opendatasci._tui.chat.commands — pure string-formatting logic."""

from opendatasci._tui.chat.commands import (
    SLASH_COMMAND_DESCRIPTIONS,
    SLASH_COMMANDS,
    format_help_message,
    format_missing_api_key_message,
)

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
        assert "##" in msg

    def test_exit_command_described(self) -> None:
        msg = format_help_message()
        assert "/exit" in msg

    def test_clear_command_described(self) -> None:
        msg = format_help_message()
        assert "/clear" in msg

    def test_config_command_described(self) -> None:
        msg = format_help_message()
        assert "/config" in msg

    def test_models_command_described(self) -> None:
        msg = format_help_message()
        assert "/models" in msg

    def test_providers_command_described(self) -> None:
        msg = format_help_message()
        assert "/providers" in msg

    def test_help_command_described(self) -> None:
        msg = format_help_message()
        assert "/help" in msg


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

    def test_config_command_registered(self) -> None:
        assert "/config" in SLASH_COMMANDS

    def test_models_command_registered(self) -> None:
        assert "/models" in SLASH_COMMANDS

    def test_providers_command_registered(self) -> None:
        assert "/providers" in SLASH_COMMANDS


# ---------------------------------------------------------------------------
# format_missing_api_key_message
# ---------------------------------------------------------------------------


class TestFormatMissingApiKeyMessage:
    def test_contains_provider_display_name(self) -> None:
        msg = format_missing_api_key_message("openai", "openai_api_key")
        assert "OpenAI" in msg

    def test_contains_env_var_name(self) -> None:
        msg = format_missing_api_key_message("openai", "openai_api_key")
        assert "OPENAI_API_KEY" in msg
