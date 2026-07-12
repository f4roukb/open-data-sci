"""Unit tests for opendatasci.session.session_manager."""

import json
import uuid
from pathlib import Path

import pytest

from opendatasci.context.local import OPENDATASCI_DIRNAME
from opendatasci.session import LocalSessionManager, SessionThread

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def manager(tmp_path: Path) -> LocalSessionManager:
    return LocalSessionManager(workspace_path=tmp_path, session_id="sess01")


def _session_file(tmp_path: Path) -> Path:
    return tmp_path / OPENDATASCI_DIRNAME / "session.json"


# ---------------------------------------------------------------------------
# LocalSessionManager
# ---------------------------------------------------------------------------


class TestLocalSessionManager:
    def test_get_current_thread_raises_when_no_threads(self, manager: LocalSessionManager) -> None:
        with pytest.raises(LookupError):
            manager.get_current_thread()

    def test_create_thread_returns_uuid(self, manager: LocalSessionManager) -> None:
        assert isinstance(manager.create_thread(), uuid.UUID)

    def test_created_thread_becomes_current(self, manager: LocalSessionManager) -> None:
        thread_id = manager.create_thread()
        assert manager.get_current_thread() == thread_id

    def test_create_thread_appends_to_existing_threads(
        self, manager: LocalSessionManager, tmp_path: Path
    ) -> None:
        first = manager.create_thread()
        second = manager.create_thread()
        assert first != second
        data = json.loads(_session_file(tmp_path).read_text(encoding="utf-8"))
        stored = [SessionThread.model_validate(t) for t in data["sess01"]["threads"]]
        assert [t.thread_id for t in stored] == [first, second]

    def test_threads_are_stored_as_thread_objects(
        self, manager: LocalSessionManager, tmp_path: Path
    ) -> None:
        manager.create_thread()
        data = json.loads(_session_file(tmp_path).read_text(encoding="utf-8"))
        (entry,) = data["sess01"]["threads"]
        thread = SessionThread.model_validate(entry)
        assert isinstance(thread.thread_id, uuid.UUID)
        assert thread.created_at.tzinfo is not None

    def test_session_entry_stamps_created_at_and_last_updated_at(
        self, manager: LocalSessionManager, tmp_path: Path
    ) -> None:
        manager.create_thread()
        data = json.loads(_session_file(tmp_path).read_text(encoding="utf-8"))
        session = data["sess01"]
        assert session["created_at"] == session["last_updated_at"]

    def test_created_at_is_stable_while_last_updated_at_advances(
        self, manager: LocalSessionManager, tmp_path: Path
    ) -> None:
        manager.create_thread()
        first = json.loads(_session_file(tmp_path).read_text(encoding="utf-8"))["sess01"]
        manager.create_thread()
        second = json.loads(_session_file(tmp_path).read_text(encoding="utf-8"))["sess01"]
        assert second["created_at"] == first["created_at"]
        assert second["last_updated_at"] >= first["last_updated_at"]
        assert len(second["threads"]) == 2

    def test_get_or_create_thread_creates_first_thread(
        self, manager: LocalSessionManager
    ) -> None:
        thread_id = manager.get_or_create_thread()
        assert manager.get_current_thread() == thread_id

    def test_get_or_create_thread_reuses_existing_thread(
        self, manager: LocalSessionManager
    ) -> None:
        first = manager.get_or_create_thread()
        assert manager.get_or_create_thread() == first

    def test_state_persists_across_instances(self, tmp_path: Path) -> None:
        writer = LocalSessionManager(workspace_path=tmp_path, session_id="sess01")
        thread_id = writer.create_thread()

        fresh = LocalSessionManager(workspace_path=tmp_path, session_id="sess01")
        assert fresh.get_current_thread() == thread_id

    def test_sessions_are_tracked_independently(self, tmp_path: Path) -> None:
        mine = LocalSessionManager(workspace_path=tmp_path, session_id="sess01")
        other = LocalSessionManager(workspace_path=tmp_path, session_id="sess02")
        mine_thread = mine.create_thread()
        other_thread = other.create_thread()
        assert mine.get_current_thread() == mine_thread
        assert other.get_current_thread() == other_thread

    def test_session_file_lives_under_opendatasci_dir(
        self, manager: LocalSessionManager, tmp_path: Path
    ) -> None:
        manager.create_thread()
        assert _session_file(tmp_path).exists()

    def test_corrupt_session_file_treated_as_empty(
        self, manager: LocalSessionManager, tmp_path: Path
    ) -> None:
        session_file = _session_file(tmp_path)
        session_file.parent.mkdir(parents=True)
        session_file.write_text("not json", encoding="utf-8")
        with pytest.raises(LookupError):
            manager.get_current_thread()
        thread_id = manager.create_thread()  # recovers by rewriting the file
        assert manager.get_current_thread() == thread_id
