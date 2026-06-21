"""CLISessionInfo: boot-time metadata consumed only by CLIController."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from opendatasci._utils.data_formats import ALL_SUPPORTED_EXTENSIONS
from opendatasci.configs import OpenDataSciConfig


class CLISessionInfo(BaseModel):
    """Metadata about a loaded session."""

    path: str
    is_directory: bool
    workspace_count: int
    workspaces: list[dict[str, Any]]
    provider: str
    model: str | None

    @classmethod
    def from_path(
        cls,
        path: str,
        workspace_path: Path | None,
        config: OpenDataSciConfig,
    ) -> "CLISessionInfo":
        file_path = Path(path)
        is_dir = file_path.is_dir()

        if is_dir and workspace_path is not None:
            data_files = [
                f
                for f in workspace_path.rglob("*")
                if f.is_file()
                and f.suffix in ALL_SUPPORTED_EXTENSIONS
                and not any(part.startswith(".") for part in f.relative_to(workspace_path).parts)
            ]
            workspaces = [{"name": f.name} for f in data_files]
            workspace_count = len(data_files)
        else:
            workspaces = [{"name": file_path.name}]
            workspace_count = 1

        return cls(
            path=path,
            is_directory=is_dir,
            workspace_count=workspace_count,
            workspaces=workspaces,
            provider=config.provider,
            model=config.model,
        )
