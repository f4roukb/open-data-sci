"""render_image — point the TUI at a static image file to display inline."""

from pathlib import Path
from typing import Any, Literal, override

from langchain_core.tools import BaseTool
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from opendatasci.streaming.events import IMAGE_ARTIFACT_KIND
from opendatasci.tools.base import OpenDataSciBaseTool

# Static-image formats only. GIF is explicitly excluded (animated content has
# no single-frame meaning here); other multi-frame/video containers are
# rejected because Pillow simply won't recognize them as a still image.
_DISALLOWED_FORMATS = frozenset({"GIF"})


class RenderImageTool(OpenDataSciBaseTool):
    """Display a static image already on disk inline in the conversation.

    Only the resolved path and caption ever leave this tool — the image
    bytes are never read into the tool's return value or the LLM's context.
    The TUI independently loads and renders the file from that path.
    """

    class CallArgs(BaseModel):
        artifact_path: str
        caption: str = ""
        summary: str
        communication: str

    name: str = "render_image"
    description: str = """\
Display a static image file from the workspace inline in the conversation.

# How to use this tool
- Point at an image file already saved to the workspace (e.g. a chart saved
  with ``execute_python_code``).
- Any static image format is supported (PNG, JPEG, BMP, WEBP, TIFF, ...).

# How NOT to use this tool
- Animated GIFs and videos are not supported — save a single still frame instead.
- Don't use this to inspect an image's content; you cannot see the rendered
  result yourself, only the user does.

Args:
    artifact_path: Path to the image file, relative to the workspace root
                   (or absolute, as long as it stays within the workspace).
    caption:       Optional short caption shown under the image.
    summary:       3-4 word status label (e.g. "Displaying churn chart").
    communication: Brief message to the user about what you're doing
                   (e.g. "Here's the chart I generated.").\
"""
    args_schema: type[BaseModel] = CallArgs
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"

    workspace_path: Path | None

    def _resolve_in_workspace(self, artifact_path: str) -> Path:
        """Resolve *artifact_path* against the workspace root, rejecting escapes."""
        if self.workspace_path is None:
            raise ValueError("No active workspace.")
        workspace_root = self.workspace_path.resolve()
        candidate = Path(artifact_path)
        resolved = (
            candidate.resolve()
            if candidate.is_absolute()
            else (workspace_root / candidate).resolve()
        )
        if not resolved.is_relative_to(workspace_root):
            raise ValueError(f"'{artifact_path}' is outside the workspace.")
        return resolved

    @staticmethod
    def _validate_static_image(path: Path) -> None:
        """Verify *path* is a real, static (non-GIF) image without reading its pixels
        into memory beyond what Pillow needs to check the format/integrity."""
        try:
            with Image.open(path) as img:
                image_format = img.format
                img.verify()
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError(f"'{path.name}' is not a valid image file.") from exc
        if image_format in _DISALLOWED_FORMATS:
            raise ValueError(
                f"'{path.name}' is a {image_format} — only static images are supported "
                "(no animated GIFs or video)."
            )

    @override
    async def _arun(
        self,
        artifact_path: str,
        summary: str,
        communication: str,
        caption: str = "",
        **kwargs: Any,
    ) -> tuple[str, dict[str, str] | None]:
        try:
            resolved = self._resolve_in_workspace(artifact_path)
            if not resolved.is_file():
                raise ValueError(f"'{artifact_path}' does not exist.")
            self._validate_static_image(resolved)
        except ValueError as exc:
            return str(exc), None

        return (
            f"Displayed image: {resolved}",
            {"kind": IMAGE_ARTIFACT_KIND, "path": str(resolved), "caption": caption},
        )


def create_media_tools(workspace_path: Path | None) -> list[BaseTool]:
    """Return media tools bound to *workspace_path*: render_image."""
    return [RenderImageTool(workspace_path=workspace_path)]
