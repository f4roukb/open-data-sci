"""Unit tests for TUI command validation in opendatasci.sandbox — validate_cli_command
and the ALLOWED_CLI_COMMANDS registry."""


import pytest

from opendatasci.sandbox.base import ALLOWED_CLI_COMMANDS, validate_cli_command

# ---------------------------------------------------------------------------
# validate_cli_command — allow / deny
# ---------------------------------------------------------------------------


class TestValidateCliCommandEmpty:
    def test_empty_string_rejected(self) -> None:
        assert validate_cli_command("") == "Empty command."

    def test_whitespace_only_rejected(self) -> None:
        assert validate_cli_command("   \t  ") == "Empty command."


class TestValidateCliCommandAllowlist:
    @pytest.mark.parametrize("cmd", ["ls -la", "pwd", "cat README.md", "grep foo bar.txt"])
    def test_allowed_simple_commands(self, cmd: str) -> None:
        assert validate_cli_command(cmd) is None

    def test_disallowed_command_rejected(self) -> None:
        result = validate_cli_command("rm -rf /tmp")
        assert result is not None
        assert "'rm'" in result
        assert "not allowed" in result

    def test_unknown_command_error_lists_some_permitted(self) -> None:
        result = validate_cli_command("doesnotexist")
        assert result is not None
        assert "Permitted commands include" in result

    def test_command_basename_only_validated(self) -> None:
        # ``/usr/bin/ls`` and ``ls`` must both be accepted (basename match).
        assert validate_cli_command("/usr/bin/ls") is None

    def test_command_name_case_insensitive(self) -> None:
        # Allowlist match is case-folded so ``LS`` resolves to ``ls``.
        assert validate_cli_command("LS") is None


class TestValidateCliCommandForbiddenOperators:
    @pytest.mark.parametrize("op", ["||", ";", "`", "$(", ">", "<"])
    def test_forbidden_operators_rejected(self, op: str) -> None:
        result = validate_cli_command(f"ls {op} echo hi")
        assert result is not None
        assert "not allowed" in result
        assert op in result


class TestValidateCliCommandPipelines:
    def test_pipe_segments_each_validated(self) -> None:
        # Both sides of a pipe must be in the allowlist.
        assert validate_cli_command("ls | grep foo") is None

    def test_pipe_with_disallowed_command_rejected(self) -> None:
        # ``rm`` is not in the allowlist; pipeline must reject.
        result = validate_cli_command("ls | rm foo")
        assert result is not None
        assert "'rm'" in result

    def test_and_chain_segments_each_validated(self) -> None:
        assert validate_cli_command("pwd && ls") is None

    def test_empty_pipeline_segment_rejected(self) -> None:
        # ``ls |`` leaves an empty trailing segment, which is invalid.
        result = validate_cli_command("ls |")
        assert result is not None
        assert "Empty pipeline segment" in result


class TestValidateCliCommandShlexErrors:
    def test_unbalanced_quote_reports_syntax_error(self) -> None:
        result = validate_cli_command('echo "unterminated')
        assert result is not None
        assert "Invalid command syntax" in result


class TestAllowedCliCommandsRegistry:
    def test_registry_is_frozenset(self) -> None:
        # Immutability matters because the value is module-level shared state.
        assert isinstance(ALLOWED_CLI_COMMANDS, frozenset)

    @pytest.mark.parametrize("cmd", ["ls", "grep", "cat", "jq"])
    def test_core_commands_present(self, cmd: str) -> None:
        assert cmd in ALLOWED_CLI_COMMANDS
