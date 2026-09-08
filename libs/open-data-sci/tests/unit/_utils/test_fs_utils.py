"""Unit tests for opendatasci._utils.fs_utils."""

import os
from pathlib import Path

import pytest

from opendatasci._utils import fs_utils
from opendatasci._utils.fs_utils import (
    _corridor_deny_paths,
    _is_within,
    find_maybe_sensitive_paths,
)


class TestIsWithin:
    def test_same_path_is_within(self) -> None:
        assert _is_within("/home/user/.ssh", "/home/user/.ssh")

    def test_child_path_is_within(self) -> None:
        parent = os.path.join("home", "user", ".ssh")
        child = os.path.join(parent, "id_rsa")
        assert _is_within(parent, child)

    def test_unrelated_path_is_not_within(self) -> None:
        parent = os.path.join("home", "user", ".ssh")
        other = os.path.join("home", "user", ".aws")
        assert not _is_within(parent, other)

    def test_sibling_with_shared_prefix_is_not_within(self) -> None:
        # ".ssh" is a string-prefix of ".ssh-agent" but not a path ancestor of it.
        parent = os.path.join("home", "user", ".ssh")
        sibling = os.path.join("home", "user", ".ssh-agent")
        assert not _is_within(parent, sibling)


class TestCorridorDenyPaths:
    def test_denies_siblings_along_path_to_keep_target(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        keep = root / "a" / "keep"
        keep.mkdir(parents=True)
        (root / "a" / "sibling1").mkdir()
        (root / "sibling2").mkdir()

        denied = _corridor_deny_paths(str(root), [str(keep)])

        assert str(root / "sibling2") in denied
        assert str(root / "a" / "sibling1") in denied
        assert str(keep) not in denied
        assert str(root / "a") not in denied
        assert str(root) not in denied

    def test_keeps_children_of_keep_target_readable(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        keep = root / "keep"
        keep.mkdir(parents=True)
        (keep / "nested").mkdir()
        (root / "sibling").mkdir()

        denied = _corridor_deny_paths(str(root), [str(keep)])

        assert str(keep / "nested") not in denied
        assert str(root / "sibling") in denied

    def test_ignores_keep_paths_outside_root(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        outside = tmp_path / "elsewhere"
        root.mkdir()
        outside.mkdir()
        (root / "child").mkdir()

        denied = _corridor_deny_paths(str(root), [str(outside)])

        # No keep target is under root, so root is denied outright (no need to
        # recurse into its children).
        assert denied == [str(root)]

    def test_multiple_keep_targets_each_get_a_corridor(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        keep_a = root / "a" / "keep"
        keep_b = root / "b" / "keep"
        keep_a.mkdir(parents=True)
        keep_b.mkdir(parents=True)
        (root / "a" / "sibling").mkdir()

        denied = _corridor_deny_paths(str(root), [str(keep_a), str(keep_b)])

        assert str(keep_a) not in denied
        assert str(keep_b) not in denied
        assert str(root / "a" / "sibling") in denied


class TestFindMaybeSensitivePaths:
    @pytest.fixture(autouse=True)
    def _fake_home(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setattr(fs_utils.os.path, "expanduser", lambda p: str(home))
        # Keep the interpreter paths well outside the fake home by default, so
        # every dotdir is denied outright unless a test overrides this.
        monkeypatch.setattr(
            fs_utils,
            "_interpreter_startup_paths",
            lambda: [str(tmp_path / "interpreter")],
        )
        return home

    def test_denies_top_level_dotdirs(self, _fake_home: Path) -> None:
        (_fake_home / ".ssh").mkdir()
        (_fake_home / ".aws").mkdir()

        denied = find_maybe_sensitive_paths()

        assert os.path.realpath(str(_fake_home / ".ssh")) in denied
        assert os.path.realpath(str(_fake_home / ".aws")) in denied

    def test_ignores_non_dotfiles(self, _fake_home: Path) -> None:
        (_fake_home / "projects").mkdir()

        denied = find_maybe_sensitive_paths()

        assert os.path.realpath(str(_fake_home / "projects")) not in denied

    def test_ignores_nested_dotfiles(self, _fake_home: Path) -> None:
        nested = _fake_home / "projects" / ".env"
        nested.parent.mkdir()
        nested.write_text("SECRET=1")

        denied = find_maybe_sensitive_paths()

        assert os.path.realpath(str(nested)) not in denied

    def test_skips_nonexistent_candidates(self, _fake_home: Path) -> None:
        # No dotfiles created, and macOS-only candidates don't exist either.
        assert find_maybe_sensitive_paths() == []

    def test_carves_corridor_when_dotdir_is_interpreter_ancestor(
        self, _fake_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pyenv = _fake_home / ".pyenv"
        interpreter_dir = pyenv / "versions" / "3.11" / "bin"
        interpreter_dir.mkdir(parents=True)
        sibling = pyenv / "versions" / "3.11" / "other"
        sibling.mkdir()
        unrelated_sibling = pyenv / "shims"
        unrelated_sibling.mkdir()

        monkeypatch.setattr(
            fs_utils,
            "_interpreter_startup_paths",
            lambda: [str(interpreter_dir)],
        )

        denied = find_maybe_sensitive_paths()

        # The interpreter directory itself must stay readable...
        assert os.path.realpath(str(interpreter_dir)) not in denied
        # ...but everything else under the dotdir stays denied.
        assert os.path.realpath(str(sibling)) in denied
        assert os.path.realpath(str(unrelated_sibling)) in denied
        # The whole dotdir must not be exempted wholesale.
        assert os.path.realpath(str(pyenv)) not in denied

    def test_unrelated_dotdir_denied_outright_even_with_interpreter_elsewhere(
        self, _fake_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (_fake_home / ".ssh").mkdir()
        interpreter_dir = _fake_home.parent / "interpreter"
        interpreter_dir.mkdir()
        monkeypatch.setattr(
            fs_utils,
            "_interpreter_startup_paths",
            lambda: [str(interpreter_dir)],
        )

        denied = find_maybe_sensitive_paths()

        assert os.path.realpath(str(_fake_home / ".ssh")) in denied
