"""LocalWorkspace — filesystem-backed workspace container."""

from pathlib import Path

from opendatasci.workspace.base import BaseWorkspace


class LocalWorkspace(BaseWorkspace):
    """Workspace backed by a local directory.

    Pass a directory to treat the whole directory as the workspace, or pass a
    single file to treat its parent directory as the workspace.
    """

    def __init__(
        self,
        path: str | Path,
    ) -> None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Path not found: {p}")
        self._directory: Path = (p.parent if p.is_file() else p).resolve()

    def get_reference(self) -> str:
        return str(self._directory)
