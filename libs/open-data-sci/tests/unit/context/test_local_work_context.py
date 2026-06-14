"""Unit tests for LocalContextStore — workspace orchestration behaviour."""


from pathlib import Path

import pytest

from opendatasci._utils.hash_utils import hash_path
from opendatasci.context.local import (
    _NOTES_DIR,
    _PROFILES_DIR,
    OPENDATASCI_DIRNAME,
    LocalContextStore,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    p = tmp_path / "workspace"
    p.mkdir()
    return p


@pytest.fixture
def ctx(workspace: Path) -> LocalContextStore:
    return LocalContextStore(workspace)


def _write_dataset(workspace: Path, name: str = "data.csv", content: str = "a,b\n1,2\n") -> Path:
    """Write a dummy dataset file and return its absolute path."""
    p = workspace / name
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# root property
# ---------------------------------------------------------------------------


class TestRoot:
    def test_root_is_opendatasci_directory_under_workspace(
        self, ctx: LocalContextStore, workspace: Path
    ) -> None:
        assert ctx.root == workspace / OPENDATASCI_DIRNAME

    async def test_root_is_created_on_session_entry(
        self, ctx: LocalContextStore, workspace: Path
    ) -> None:
        assert not (workspace / OPENDATASCI_DIRNAME).exists()
        async with ctx.session():
            assert (workspace / OPENDATASCI_DIRNAME).exists()

    async def test_root_is_a_directory_after_session_entry(
        self, ctx: LocalContextStore
    ) -> None:
        async with ctx.session():
            assert ctx.root.is_dir()

    def test_root_is_idempotent(self, ctx: LocalContextStore) -> None:
        r1 = ctx.root
        r2 = ctx.root
        assert r1 == r2

    async def test_root_creation_is_idempotent(self, ctx: LocalContextStore) -> None:
        async with ctx.session():
            pass
        async with ctx.session():  # second entry must not raise
            pass


# ---------------------------------------------------------------------------
# Storage layout
# ---------------------------------------------------------------------------


class TestStorageLayout:
    def test_notes_root_under_opendatasci_dataset_notes(
        self, ctx: LocalContextStore, workspace: Path
    ) -> None:
        assert ctx._notes_root == workspace / OPENDATASCI_DIRNAME / _NOTES_DIR

    def test_profiles_root_under_opendatasci_dataset_profiling(
        self, ctx: LocalContextStore, workspace: Path
    ) -> None:
        assert ctx._profiles_root == workspace / OPENDATASCI_DIRNAME / _PROFILES_DIR


# ---------------------------------------------------------------------------
# read_dataset_info
# ---------------------------------------------------------------------------


class TestReadDatasetInfo:
    async def test_raises_for_missing_dataset(self, ctx: LocalContextStore, workspace: Path) -> None:
        with pytest.raises(FileNotFoundError):
            await ctx.read_dataset_info(str(workspace / "nonexistent.csv"))

    async def test_returns_string(self, ctx: LocalContextStore, workspace: Path) -> None:
        p = _write_dataset(workspace)
        result = await ctx.read_dataset_info(str(p))
        assert isinstance(result, str)

    async def test_default_content_contains_dataset_name(
        self, ctx: LocalContextStore, workspace: Path
    ) -> None:
        p = _write_dataset(workspace, "report.csv")
        result = await ctx.read_dataset_info(str(p))
        assert "report.csv" in result

    async def test_default_content_contains_placeholder(
        self, ctx: LocalContextStore, workspace: Path
    ) -> None:
        p = _write_dataset(workspace)
        result = await ctx.read_dataset_info(str(p))
        assert "No notes recorded yet" in result

    async def test_default_content_has_notes_header(
        self, ctx: LocalContextStore, workspace: Path
    ) -> None:
        p = _write_dataset(workspace)
        result = await ctx.read_dataset_info(str(p))
        assert "# DATASET NOTES" in result

    async def test_existing_notes_are_returned(
        self, ctx: LocalContextStore, workspace: Path
    ) -> None:
        p = _write_dataset(workspace)
        await ctx.update_dataset_info(str(p), "# My Notes", merge=False)
        result = await ctx.read_dataset_info(str(p))
        assert "# My Notes" in result

    async def test_profile_section_prepended_when_present(
        self, ctx: LocalContextStore, workspace: Path
    ) -> None:
        p = _write_dataset(workspace)
        hash_hex = await hash_path(p)
        await ctx.update_dataset_info(str(p), "notes text", merge=False)
        ctx.save_dataset_profile(hash_hex, "profile content")
        result = await ctx.read_dataset_info(str(p))
        assert result.index("# DATASET PROFILING") < result.index("# DATASET NOTES")
        assert "profile content" in result
        assert "notes text" in result

    async def test_profiling_header_present_when_profile_exists(
        self, ctx: LocalContextStore, workspace: Path
    ) -> None:
        p = _write_dataset(workspace)
        hash_hex = await hash_path(p)
        await ctx.update_dataset_info(str(p), "notes", merge=False)
        ctx.save_dataset_profile(hash_hex, "profile")
        result = await ctx.read_dataset_info(str(p))
        assert "# DATASET PROFILING" in result

    async def test_separator_present_when_profile_exists(
        self, ctx: LocalContextStore, workspace: Path
    ) -> None:
        p = _write_dataset(workspace)
        hash_hex = await hash_path(p)
        await ctx.update_dataset_info(str(p), "notes", merge=False)
        ctx.save_dataset_profile(hash_hex, "profile")
        result = await ctx.read_dataset_info(str(p))
        assert "---" in result

    async def test_no_profiling_header_when_profile_absent(
        self, ctx: LocalContextStore, workspace: Path
    ) -> None:
        p = _write_dataset(workspace)
        await ctx.update_dataset_info(str(p), "notes", merge=False)
        result = await ctx.read_dataset_info(str(p))
        assert "# DATASET PROFILING" not in result
        assert "---" not in result

    async def test_no_file_written_to_disk(self, ctx: LocalContextStore, workspace: Path) -> None:
        p = _write_dataset(workspace)
        await ctx.read_dataset_info(str(p))
        assert not (workspace / "context").exists()

    async def test_accepts_relative_dataset_path(
        self, ctx: LocalContextStore, workspace: Path
    ) -> None:
        _write_dataset(workspace, "rel.csv")
        result = await ctx.read_dataset_info("rel.csv")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# update_dataset_info
# ---------------------------------------------------------------------------


class TestUpdateDatasetInfo:
    async def test_raises_for_missing_dataset(self, ctx: LocalContextStore, workspace: Path) -> None:
        with pytest.raises(FileNotFoundError):
            await ctx.update_dataset_info(str(workspace / "ghost.csv"), "text")

    async def test_returns_string_path(self, ctx: LocalContextStore, workspace: Path) -> None:
        p = _write_dataset(workspace)
        result = await ctx.update_dataset_info(str(p), "some notes")
        assert isinstance(result, str)

    async def test_returned_path_is_inside_dataset_notes_dir(
        self, ctx: LocalContextStore, workspace: Path
    ) -> None:
        p = _write_dataset(workspace)
        result = await ctx.update_dataset_info(str(p), "some notes")
        assert _NOTES_DIR in result

    async def test_notes_are_persisted(
        self, ctx: LocalContextStore, workspace: Path
    ) -> None:
        p = _write_dataset(workspace)
        await ctx.update_dataset_info(str(p), "my note")
        result = await ctx.read_dataset_info(str(p))
        assert "my note" in result

    async def test_merge_true_appends_to_existing(
        self, ctx: LocalContextStore, workspace: Path
    ) -> None:
        p = _write_dataset(workspace)
        await ctx.update_dataset_info(str(p), "first entry", merge=False)
        await ctx.update_dataset_info(str(p), "second entry", merge=True)
        result = await ctx.read_dataset_info(str(p))
        assert "first entry" in result
        assert "second entry" in result

    async def test_merge_false_replaces_content(
        self, ctx: LocalContextStore, workspace: Path
    ) -> None:
        p = _write_dataset(workspace)
        await ctx.update_dataset_info(str(p), "first", merge=True)
        await ctx.update_dataset_info(str(p), "replacement", merge=False)
        result = await ctx.read_dataset_info(str(p))
        assert "replacement" in result
        assert "first" not in result

    async def test_merge_with_no_existing_uses_only_update(
        self, ctx: LocalContextStore, workspace: Path
    ) -> None:
        p = _write_dataset(workspace)
        await ctx.update_dataset_info(str(p), "fresh note", merge=True)
        result = await ctx.read_dataset_info(str(p))
        assert "fresh note" in result

    async def test_update_is_stripped_before_saving(
        self, ctx: LocalContextStore, workspace: Path
    ) -> None:
        p = _write_dataset(workspace)
        await ctx.update_dataset_info(str(p), "  padded  ", merge=False)
        notes = ctx._load_notes(await hash_path(p))
        assert notes == "padded\n"

    async def test_saves_with_correct_hash(
        self, ctx: LocalContextStore, workspace: Path
    ) -> None:
        p = _write_dataset(workspace)
        await ctx.update_dataset_info(str(p), "note")
        hash_hex = await hash_path(p)
        assert ctx._load_notes(hash_hex) is not None

    async def test_does_not_write_profile(
        self, ctx: LocalContextStore, workspace: Path
    ) -> None:
        p = _write_dataset(workspace)
        await ctx.update_dataset_info(str(p), "note")
        hash_hex = await hash_path(p)
        _, _, profile = await ctx.get_profile_info(str(p))
        assert profile is None

    async def test_no_mirror_file_written(self, ctx: LocalContextStore, workspace: Path) -> None:
        p = _write_dataset(workspace)
        await ctx.update_dataset_info(str(p), "note")
        assert not (workspace / "context").exists()


# ---------------------------------------------------------------------------
# get_profile_info
# ---------------------------------------------------------------------------


class TestGetProfileInfo:
    async def test_raises_for_missing_dataset(self, ctx: LocalContextStore, workspace: Path) -> None:
        with pytest.raises(FileNotFoundError):
            await ctx.get_profile_info(str(workspace / "nope.csv"))

    async def test_returns_three_tuple(self, ctx: LocalContextStore, workspace: Path) -> None:
        p = _write_dataset(workspace)
        result = await ctx.get_profile_info(str(p))
        assert len(result) == 3

    async def test_first_element_is_resolved_path(
        self, ctx: LocalContextStore, workspace: Path
    ) -> None:
        p = _write_dataset(workspace)
        resolved, _, _ = await ctx.get_profile_info(str(p))
        assert Path(resolved) == p.resolve()

    async def test_second_element_is_string_hash(
        self, ctx: LocalContextStore, workspace: Path
    ) -> None:
        p = _write_dataset(workspace)
        _, hash_hex, _ = await ctx.get_profile_info(str(p))
        assert isinstance(hash_hex, str)
        assert len(hash_hex) > 0

    async def test_third_element_is_none_when_no_profile(
        self, ctx: LocalContextStore, workspace: Path
    ) -> None:
        p = _write_dataset(workspace)
        _, _, existing = await ctx.get_profile_info(str(p))
        assert existing is None

    async def test_third_element_returns_existing_profile(
        self, ctx: LocalContextStore, workspace: Path
    ) -> None:
        p = _write_dataset(workspace)
        hash_hex = await hash_path(p)
        ctx.save_dataset_profile(hash_hex, "# Cached")
        _, _, existing = await ctx.get_profile_info(str(p))
        assert existing == "# Cached"

    async def test_same_file_produces_same_hash(
        self, ctx: LocalContextStore, workspace: Path
    ) -> None:
        p = _write_dataset(workspace)
        _, hash1, _ = await ctx.get_profile_info(str(p))
        _, hash2, _ = await ctx.get_profile_info(str(p))
        assert hash1 == hash2

    async def test_different_files_produce_different_hashes(
        self, ctx: LocalContextStore, workspace: Path
    ) -> None:
        p1 = _write_dataset(workspace, "a.csv", "1,2\n")
        p2 = _write_dataset(workspace, "b.csv", "3,4\n")
        _, h1, _ = await ctx.get_profile_info(str(p1))
        _, h2, _ = await ctx.get_profile_info(str(p2))
        assert h1 != h2


# ---------------------------------------------------------------------------
# save_dataset_profile
# ---------------------------------------------------------------------------


class TestSaveDatasetProfile:
    def test_profile_is_retrievable_after_save(self, ctx: LocalContextStore) -> None:
        ctx.save_dataset_profile("abc", "content")
        assert ctx._load_profile("abc") == "content"

    def test_save_profile_persists_to_disk(self, ctx: LocalContextStore, workspace: Path) -> None:
        ctx.save_dataset_profile("myhash", "# Card")
        expected = workspace / OPENDATASCI_DIRNAME / _PROFILES_DIR / "myhash.md"
        assert expected.read_text(encoding="utf-8") == "# Card"
