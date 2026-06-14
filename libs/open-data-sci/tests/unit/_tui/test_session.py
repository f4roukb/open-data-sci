"""Unit tests for opendatasci._tui.session.CLISessionInfo."""


import pytest
from pydantic import ValidationError

from opendatasci._tui.session import CLISessionInfo


def _make_session_info(**overrides: object) -> CLISessionInfo:
    defaults: dict[str, object] = {
        "path": "/data/report.csv",
        "is_directory": False,
        "workspace_count": 1,
        "workspaces": [{"name": "report.csv"}],
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
    }
    defaults.update(overrides)
    return CLISessionInfo(**defaults)  # type: ignore[arg-type]


class TestCLISessionInfo:
    def test_required_fields_are_stored(self) -> None:
        info = _make_session_info()

        assert info.path == "/data/report.csv"
        assert info.is_directory is False
        assert info.workspace_count == 1
        assert info.workspaces == [{"name": "report.csv"}]
        assert info.provider == "anthropic"
        assert info.model == "claude-sonnet-4-6"

    def test_model_accepts_none(self) -> None:
        info = _make_session_info(model=None)

        assert info.model is None

    def test_model_dump_returns_all_fields(self) -> None:
        info = _make_session_info()
        dumped = info.model_dump()

        assert set(dumped.keys()) == {
            "path",
            "is_directory",
            "workspace_count",
            "workspaces",
            "provider",
            "model",
        }

    def test_model_dump_preserves_values(self) -> None:
        info = _make_session_info()
        dumped = info.model_dump()

        assert dumped["path"] == info.path
        assert dumped["is_directory"] == info.is_directory
        assert dumped["workspace_count"] == info.workspace_count
        assert dumped["workspaces"] == info.workspaces
        assert dumped["provider"] == info.provider
        assert dumped["model"] == info.model

    def test_invalid_workspace_count_type_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            _make_session_info(workspace_count="not-an-int")  # type: ignore[arg-type]

    def test_is_directory_true_for_directory_session(self) -> None:
        info = _make_session_info(
            is_directory=True,
            workspace_count=3,
            workspaces=[
                {"name": "a.csv"},
                {"name": "b.xlsx"},
                {"name": "c.csv"},
            ],
        )

        assert info.is_directory is True
        assert info.workspace_count == 3
        assert len(info.workspaces) == 3
