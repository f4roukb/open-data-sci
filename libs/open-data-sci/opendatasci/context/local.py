"""File-based context stores backed by the workspace's ``.opendatasci`` directory."""

import datetime
import logging
from contextlib import asynccontextmanager
from datetime import datetime as _datetime
from datetime import timezone
from pathlib import Path
from typing import AsyncGenerator, Self

from opendatasci._utils.hash_utils import hash_path
from opendatasci.context.base import BaseContextStore

logger = logging.getLogger(__name__)

OPENDATASCI_DIRNAME = ".opendatasci"

_NOTES_DIR = "dataset_notes"
_PROFILES_DIR = "dataset_profiling"
_PLANS_DIR = "plans"


class LocalContextStore(BaseContextStore):
    """File-based context store for a single local workspace.

    Persists dataset notes and profile cards (keyed by dataset path) as well as
    session plans (keyed by ``session_id``).  All data lives under the
    workspace's ``.opendatasci`` directory.

    Args:
        workspace_path: Root directory of the active workspace.  Relative
            dataset paths are resolved against this directory.
    """

    def __init__(self, workspace_path: Path) -> None:
        self._workspace_path = workspace_path
        self._root = workspace_path / OPENDATASCI_DIRNAME
        self._current_plan: str | None = None

    # ── BaseContextStore: session ────────────────────────────────────

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[Self, None]:
        self._root.mkdir(parents=True, exist_ok=True)
        yield self

    # ── BaseContextStore: root ──────────────────────────────────────

    @property
    def root(self) -> Path:
        return self._root

    # ── Internal storage paths ───────────────────────────────────────────────

    @property
    def _notes_root(self) -> Path:
        return self.root / _NOTES_DIR

    @property
    def _profiles_root(self) -> Path:
        return self.root / _PROFILES_DIR

    @property
    def _plans_root(self) -> Path:
        return self.root / _PLANS_DIR

    # ── Internal notes storage ────────────────────────────────────────────────

    def _find(self, hash_hex: str) -> Path | None:
        """Locate an existing notes file for *hash_hex* under any date prefix."""
        if self._notes_root.exists():
            for p in self._notes_root.rglob(f"{hash_hex}.md"):
                return p
        return None

    def _resolve_notes_path(self, hash_hex: str) -> Path:
        """Reuse an existing file for *hash_hex*, or build a new date-keyed path."""
        existing = self._find(hash_hex)
        if existing is not None:
            return existing
        today = datetime.date.today()
        return (
            self._notes_root
            / str(today.year)
            / f"{today.month:02d}"
            / f"{today.day:02d}"
            / f"{hash_hex}.md"
        )

    def _load_notes(self, hash_hex: str) -> str | None:
        p = self._find(hash_hex)
        return p.read_text(encoding="utf-8") if p is not None else None

    def _save_notes(self, hash_hex: str, content: str) -> None:
        path = self._resolve_notes_path(hash_hex)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def _notes_file_path(self, hash_hex: str) -> str:
        return str(self._resolve_notes_path(hash_hex))

    # ── Internal profile storage ──────────────────────────────────────────────

    def _load_profile(self, hash_hex: str) -> str | None:
        p = self._profiles_root / f"{hash_hex}.md"
        return p.read_text(encoding="utf-8") if p.exists() else None

    # ── BaseContextStore interface ──────────────────────────────────

    async def read_dataset_info(self, dataset_path: str) -> str:
        """Return combined dataset info: profile card (if any) + session notes.

        The returned Markdown string has clearly labelled sections:

        ``# DATASET PROFILING`` — auto-generated stats (shape, dtypes, etc.)
        ``# DATASET NOTES``     — persistent notes written by the agent

        If no profile exists, only the notes section is included.
        If no notes exist yet, a placeholder scaffold is returned.
        """
        path = self._resolve_dataset_path(dataset_path)
        hash_hex = await hash_path(path)

        notes = self._load_notes(hash_hex)
        if notes is None:
            notes = f"# DATASET NOTES: {path.name}\n\n_No notes recorded yet._\n"
        else:
            notes = "# DATASET NOTES\n\n" + notes

        profile = self._load_profile(hash_hex)
        if profile is not None:
            return "# DATASET PROFILING\n\n" + profile.rstrip() + "\n\n---\n\n" + notes
        return notes

    async def update_dataset_info(
        self,
        dataset_path: str,
        update: str,
        merge: bool = True,
    ) -> str:
        """Persist dataset notes and return the path to the stored notes file.

        Only notes are modified — the profile card (if any) is never touched.
        """
        path = self._resolve_dataset_path(dataset_path)
        hash_hex = await hash_path(path)

        if merge:
            existing = self._load_notes(hash_hex)
            new_content = (
                existing.rstrip() + "\n\n" + update.strip() + "\n"
                if existing
                else update.strip() + "\n"
            )
        else:
            new_content = update.strip() + "\n"

        self._save_notes(hash_hex, new_content)
        return self._notes_file_path(hash_hex)

    async def get_profile_info(self, dataset_path: str) -> tuple[str, str, str | None]:
        """Return ``(resolved_path_str, hash_hex, existing_profile_or_None)``.

        Used by the ``profile_dataset`` tool to check for a cached card before
        running the profiling sandbox pass.
        """
        path = self._resolve_dataset_path(dataset_path)
        hash_hex = await hash_path(path)
        return str(path), hash_hex, self._load_profile(hash_hex)

    def save_dataset_profile(self, hash_hex: str, content: str) -> None:
        """Persist a completed profile card for *hash_hex*."""
        self._profiles_root.mkdir(parents=True, exist_ok=True)
        (self._profiles_root / f"{hash_hex}.md").write_text(content, encoding="utf-8")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _resolve_dataset_path(self, dataset_path: str) -> Path:
        p = Path(dataset_path)
        path = (self._workspace_path / p).resolve() if not p.is_absolute() else p.resolve()
        if not path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {path}")
        return path

    # ── BaseContextStore: plans ───────────────────────────────────────────────

    def current_plan(self, session_id: str) -> str | None:
        """Return the most recent plan for this session.

        Resolves the plan by reading the latest ``{session_id}_*.txt`` file from
        disk so the plan survives process restarts.  Falls back to the in-memory
        cache when no file exists yet or a disk read fails.
        """
        if self._plans_root.exists():
            files = sorted(self._plans_root.glob(f"{session_id}_*.txt"))
            if files:
                try:
                    return files[-1].read_text(encoding="utf-8")
                except OSError:
                    logger.warning("Could not read plan file: %s", files[-1], exc_info=True)
        return self._current_plan

    def save_plan(self, session_id: str, plan: str) -> None:
        """Persist *plan* to disk and prune stale files.

        Also updates the in-memory cache used as a fallback when no plan file
        can be read back.
        """
        self._current_plan = plan
        self._plans_root.mkdir(parents=True, exist_ok=True)
        stamp = _datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = self._plans_root / f"{session_id}_{stamp}.txt"
        try:
            path.write_text(plan, encoding="utf-8")
        except OSError:
            logger.warning("Could not write plan file: %s", path, exc_info=True)
            return
        self.prune()

    def prune(self) -> None:
        """Keep only the most recent plan file per session_id."""
        if not self._plans_root.exists():
            return
        by_session: dict[str, list[Path]] = {}
        for p in self._plans_root.glob("*.txt"):
            sid = p.stem.partition("_")[0]
            if not sid:
                continue
            by_session.setdefault(sid, []).append(p)
        for files in by_session.values():
            files.sort(key=lambda f: f.name)
            for stale in files[:-1]:
                try:
                    stale.unlink()
                except OSError:
                    logger.exception("Could not delete stale plan file: %s", stale)
