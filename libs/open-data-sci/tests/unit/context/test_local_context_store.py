"""Unit tests for LocalContextStore — file-based storage internals."""


import datetime
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from opendatasci.context.local import LocalContextStore, OPENDATASCI_DIRNAME, _NOTES_DIR, _PROFILES_DIR

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> LocalContextStore:
    return LocalContextStore(tmp_path)


HASH = "abc123def456"


# ---------------------------------------------------------------------------
# _load_notes
# ---------------------------------------------------------------------------


class TestLoadNotes:
    def test_returns_none_when_notes_root_absent(self, store: LocalContextStore) -> None:
        assert store._load_notes(HASH) is None

    def test_returns_none_for_unknown_hash(self, store: LocalContextStore) -> None:
        store._notes_root.mkdir(parents=True, exist_ok=True)
        assert store._load_notes("nonexistent") is None

    def test_returns_saved_content(self, store: LocalContextStore) -> None:
        store._save_notes(HASH, "hello notes")
        assert store._load_notes(HASH) == "hello notes"

    def test_returns_exact_bytes(self, store: LocalContextStore) -> None:
        content = "line1\nline2\n\n## Section\n"
        store._save_notes(HASH, content)
        assert store._load_notes(HASH) == content

    def test_finds_note_across_date_subdirectories(
        self, store: LocalContextStore, tmp_path: Path
    ) -> None:
        nested = tmp_path / OPENDATASCI_DIRNAME / _NOTES_DIR / "2020" / "01" / "15"
        nested.mkdir(parents=True)
        (nested / f"{HASH}.md").write_text("planted", encoding="utf-8")
        assert store._load_notes(HASH) == "planted"


# ---------------------------------------------------------------------------
# _save_notes
# ---------------------------------------------------------------------------


class TestSaveNotes:
    def test_creates_notes_root_on_first_save(
        self, store: LocalContextStore, tmp_path: Path
    ) -> None:
        store._save_notes(HASH, "content")
        assert store._notes_root.exists()

    def test_file_is_placed_under_date_keyed_path(self, store: LocalContextStore) -> None:
        today = datetime.date.today()
        store._save_notes(HASH, "x")
        expected_dir = store._notes_root / str(today.year) / f"{today.month:02d}" / f"{today.day:02d}"
        assert (expected_dir / f"{HASH}.md").exists()

    def test_overwrites_existing_content(self, store: LocalContextStore) -> None:
        store._save_notes(HASH, "first")
        store._save_notes(HASH, "second")
        assert store._load_notes(HASH) == "second"

    def test_second_save_reuses_same_file_path(self, store: LocalContextStore) -> None:
        store._save_notes(HASH, "v1")
        path_after_first = store._find(HASH)
        store._save_notes(HASH, "v2")
        path_after_second = store._find(HASH)
        assert path_after_first == path_after_second

    def test_different_hashes_produce_different_files(self, store: LocalContextStore) -> None:
        store._save_notes("aaa", "a-content")
        store._save_notes("bbb", "b-content")
        assert store._load_notes("aaa") == "a-content"
        assert store._load_notes("bbb") == "b-content"

    def test_frozen_date_used_for_new_file(self, store: LocalContextStore) -> None:
        frozen = datetime.date(2000, 6, 15)
        with patch("opendatasci.context.local.datetime") as mock_dt:
            mock_dt.date.today.return_value = frozen
            store._save_notes(HASH, "dated")
        expected_dir = store._notes_root / "2000" / "06" / "15"
        assert (expected_dir / f"{HASH}.md").exists()


# ---------------------------------------------------------------------------
# _load_profile
# ---------------------------------------------------------------------------


class TestLoadProfile:
    def test_returns_none_when_absent(self, store: LocalContextStore) -> None:
        assert store._load_profile(HASH) is None

    def test_returns_saved_profile(self, store: LocalContextStore) -> None:
        store.save_dataset_profile(HASH, "# My Profile\n")
        assert store._load_profile(HASH) == "# My Profile\n"

    def test_returns_exact_content(self, store: LocalContextStore) -> None:
        content = "col|type\n---|---\nid|int\n"
        store.save_dataset_profile(HASH, content)
        assert store._load_profile(HASH) == content


# ---------------------------------------------------------------------------
# save_dataset_profile
# ---------------------------------------------------------------------------


class TestSaveDatasetProfile:
    def test_creates_profiles_root(self, store: LocalContextStore, tmp_path: Path) -> None:
        store.save_dataset_profile(HASH, "content")
        assert store._profiles_root.exists()

    def test_file_is_flat_under_profiles_root(
        self, store: LocalContextStore
    ) -> None:
        store.save_dataset_profile(HASH, "content")
        assert (store._profiles_root / f"{HASH}.md").exists()

    def test_overwrites_existing_profile(self, store: LocalContextStore) -> None:
        store.save_dataset_profile(HASH, "v1")
        store.save_dataset_profile(HASH, "v2")
        assert store._load_profile(HASH) == "v2"

    def test_different_hashes_stored_independently(self, store: LocalContextStore) -> None:
        store.save_dataset_profile("h1", "first")
        store.save_dataset_profile("h2", "second")
        assert store._load_profile("h1") == "first"
        assert store._load_profile("h2") == "second"


# ---------------------------------------------------------------------------
# LocalContextStore — plan storage
# ---------------------------------------------------------------------------


class TestLocalContextStorePlans:
    def test_initial_get_current_plan_is_none(self, store: LocalContextStore) -> None:
        assert store.get_current_plan("abc123") is None

    def test_save_plan_then_get_current_plan_returns_it(self, store: LocalContextStore) -> None:
        store.save_plan("abc123", "my plan")
        plan = store.get_current_plan("abc123")
        assert plan is not None
        assert plan.content == "my plan"

    def test_save_plan_overwrites_previous(self, store: LocalContextStore) -> None:
        store.save_plan("abc123", "plan v1")
        store.save_plan("abc123", "plan v2")
        plan = store.get_current_plan("abc123")
        assert plan is not None
        assert plan.content == "plan v2"

    def test_save_plan_stamps_created_at_metadata(self, store: LocalContextStore) -> None:
        store.save_plan("abc123", "x")
        plan = store.get_current_plan("abc123")
        assert plan is not None
        assert "created_at" in plan.metadata

    def test_prune_no_op_when_plans_dir_missing(self, store: LocalContextStore) -> None:
        store.prune()  # Must not raise

    def test_plans_live_under_opendatasci_dir(self, store: LocalContextStore, tmp_path: Path) -> None:
        store.save_plan("s1", "plan")
        assert store._plans_root == tmp_path / OPENDATASCI_DIRNAME / "plans"
        assert store._plans_root.exists()

    def test_save_plan_writes_json_file(self, store: LocalContextStore) -> None:
        store.save_plan("testsid", "my plan content")
        files = list(store._plans_root.glob("*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text(encoding="utf-8"))
        assert data["content"] == "my plan content"

    def test_save_plan_creates_directory(self, store: LocalContextStore) -> None:
        store.save_plan("s1", "plan")
        assert store._plans_root.exists()

    def test_filename_starts_with_session_id(self, store: LocalContextStore) -> None:
        store.save_plan("myid1234", "x")
        files = list(store._plans_root.glob("*.json"))
        assert files[0].name.startswith("myid1234_")

    def test_prune_keeps_only_latest_per_session(self, store: LocalContextStore) -> None:
        plans = store._plans_root
        plans.mkdir(parents=True)
        (plans / "sess01_20240101T000000Z.json").write_text(json.dumps({"content": "plan v1", "metadata": {}}))
        (plans / "sess01_20240101T000001Z.json").write_text(json.dumps({"content": "plan v2", "metadata": {}}))
        store.prune()
        files = list(plans.glob("sess01_*.json"))
        assert len(files) == 1
        assert json.loads(files[0].read_text(encoding="utf-8"))["content"] == "plan v2"

    def test_prune_preserves_other_sessions(self, store: LocalContextStore) -> None:
        plans = store._plans_root
        plans.mkdir(parents=True)
        (plans / "other123_20240101T000000Z.json").write_text(
            json.dumps({"content": "other session plan", "metadata": {}})
        )
        store.save_plan("sess01", "plan v1")
        assert (plans / "other123_20240101T000000Z.json").exists()

    def test_explicit_prune_removes_stale_files(self, store: LocalContextStore) -> None:
        plans = store._plans_root
        plans.mkdir(parents=True)
        (plans / "sess01_20240101T000000Z.json").write_text(json.dumps({"content": "old", "metadata": {}}))
        (plans / "sess01_20240101T000001Z.json").write_text(json.dumps({"content": "new", "metadata": {}}))
        store.prune()
        files = list(plans.glob("sess01_*.json"))
        assert len(files) == 1
        assert json.loads(files[0].read_text())["content"] == "new"

    def test_get_current_plan_reads_from_disk_not_memory(self, tmp_path: Path) -> None:
        writer = LocalContextStore(tmp_path)
        writer.save_plan("s1", "persisted plan")

        fresh_store = LocalContextStore(tmp_path)
        plan = fresh_store.get_current_plan("s1")
        assert plan is not None
        assert plan.content == "persisted plan"

    def test_get_current_plan_loads_from_disk_after_restart(self, tmp_path: Path) -> None:
        store = LocalContextStore(tmp_path)
        store.save_plan("s1", "persisted plan")

        fresh_store = LocalContextStore(tmp_path)
        plan = fresh_store.get_current_plan("s1")
        assert plan is not None
        assert plan.content == "persisted plan"

    def test_get_current_plan_returns_latest_of_multiple_disk_files(self, store: LocalContextStore) -> None:
        plans = store._plans_root
        plans.mkdir(parents=True)
        (plans / "s1_20240101T000000Z.json").write_text(json.dumps({"content": "old plan", "metadata": {}}))
        (plans / "s1_20240101T000001Z.json").write_text(json.dumps({"content": "new plan", "metadata": {}}))

        plan = store.get_current_plan("s1")
        assert plan is not None
        assert plan.content == "new plan"

    def test_get_current_plan_ignores_other_session_files(self, store: LocalContextStore) -> None:
        plans = store._plans_root
        plans.mkdir(parents=True)
        (plans / "othersid_20240101T000000Z.json").write_text(
            json.dumps({"content": "other session plan", "metadata": {}})
        )

        assert store.get_current_plan("mysessid") is None

    def test_get_current_plan_does_not_match_session_id_prefix(self, store: LocalContextStore) -> None:
        plans = store._plans_root
        plans.mkdir(parents=True)
        (plans / "s10_20240101T000000Z.json").write_text(json.dumps({"content": "s10 plan", "metadata": {}}))

        assert store.get_current_plan("s1") is None

    def test_get_current_plan_returns_none_when_dir_missing(self, store: LocalContextStore) -> None:
        assert store.get_current_plan("s1") is None

    def test_get_current_plan_returns_none_on_read_error(self, store: LocalContextStore) -> None:
        store.save_plan("s1", "cached plan")

        with patch("pathlib.Path.read_text", side_effect=OSError("disk error")):
            result = store.get_current_plan("s1")

        # No in-memory fallback — get_current_plan always resolves fresh from disk.
        assert result is None
