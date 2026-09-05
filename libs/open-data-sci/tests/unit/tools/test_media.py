"""Unit tests for opendatasci.tools.media."""

from pathlib import Path

from PIL import Image

from opendatasci.streaming.events import IMAGE_ARTIFACT_KIND
from opendatasci.tools.media import create_media_tools

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_tool(workspace_path: Path | None):
    tools = create_media_tools(workspace_path)
    return tools[0]


def _make_png(path: Path, size: tuple[int, int] = (4, 4), color: str = "red") -> Path:
    Image.new("RGB", size, color=color).save(path, format="PNG")
    return path


def _make_gif(path: Path) -> Path:
    Image.new("RGB", (4, 4), color="blue").save(path, format="GIF")
    return path


async def _invoke(tool, artifact_path: str, caption: str = "") -> tuple[str, dict | None]:
    """Call _arun directly to get the raw (content, artifact) tuple.

    ``ainvoke`` on a content_and_artifact tool wraps the tuple into a
    ToolMessage, discarding the artifact from the return value the test
    needs to inspect, so _arun is called directly instead.
    """
    return await tool._arun(
        artifact_path=artifact_path, caption=caption, summary="Displaying image", communication=""
    )


# ---------------------------------------------------------------------------
# create_media_tools — structure
# ---------------------------------------------------------------------------


class TestCreateMediaToolsStructure:
    def test_returns_one_tool(self) -> None:
        tools = create_media_tools(None)
        assert len(tools) == 1

    def test_tool_name_is_render_image(self) -> None:
        tools = create_media_tools(None)
        assert tools[0].name == "render_image"

    def test_response_format_is_content_and_artifact(self) -> None:
        tools = create_media_tools(None)
        assert tools[0].response_format == "content_and_artifact"


# ---------------------------------------------------------------------------
# render_image — no workspace
# ---------------------------------------------------------------------------


class TestRenderImageNoWorkspace:
    async def test_no_workspace_path_returns_error_content(self) -> None:
        tool = _get_tool(None)
        content, artifact = await _invoke(tool, "chart.png")
        assert "No active workspace" in content
        assert artifact is None


# ---------------------------------------------------------------------------
# render_image — path resolution & validation
# ---------------------------------------------------------------------------


class TestRenderImageValidation:
    async def test_missing_file_returns_error_content(self, tmp_path: Path) -> None:
        tool = _get_tool(tmp_path)
        content, artifact = await _invoke(tool, "missing.png")
        assert "does not exist" in content
        assert artifact is None

    async def test_path_traversal_outside_workspace_rejected(self, tmp_path: Path) -> None:
        outside = tmp_path.parent / "outside_secret.png"
        _make_png(outside)
        try:
            tool = _get_tool(tmp_path)
            content, artifact = await _invoke(tool, "../outside_secret.png")
            assert "outside the workspace" in content
            assert artifact is None
        finally:
            outside.unlink(missing_ok=True)

    async def test_absolute_path_outside_workspace_rejected(self, tmp_path: Path) -> None:
        other_dir = tmp_path.parent / "other_workspace"
        other_dir.mkdir(exist_ok=True)
        try:
            outside_file = _make_png(other_dir / "chart.png")
            tool = _get_tool(tmp_path)
            content, artifact = await _invoke(tool, str(outside_file))
            assert "outside the workspace" in content
            assert artifact is None
        finally:
            for f in other_dir.glob("*"):
                f.unlink()
            other_dir.rmdir()

    async def test_absolute_path_inside_workspace_accepted(self, tmp_path: Path) -> None:
        img = _make_png(tmp_path / "chart.png")
        tool = _get_tool(tmp_path)
        content, artifact = await _invoke(tool, str(img))
        assert artifact is not None
        assert artifact["path"] == str(img.resolve())

    async def test_non_image_file_rejected(self, tmp_path: Path) -> None:
        text_file = tmp_path / "notes.txt"
        text_file.write_text("hello")
        tool = _get_tool(tmp_path)
        content, artifact = await _invoke(tool, "notes.txt")
        assert "not a valid image" in content
        assert artifact is None

    async def test_corrupt_image_bytes_rejected(self, tmp_path: Path) -> None:
        fake = tmp_path / "chart.png"
        fake.write_bytes(b"not actually a png")
        tool = _get_tool(tmp_path)
        content, artifact = await _invoke(tool, "chart.png")
        assert "not a valid image" in content
        assert artifact is None

    async def test_gif_rejected(self, tmp_path: Path) -> None:
        _make_gif(tmp_path / "anim.gif")
        tool = _get_tool(tmp_path)
        content, artifact = await _invoke(tool, "anim.gif")
        assert "GIF" in content
        assert "static images" in content
        assert artifact is None

    async def test_directory_path_rejected(self, tmp_path: Path) -> None:
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        tool = _get_tool(tmp_path)
        content, artifact = await _invoke(tool, "subdir")
        assert "does not exist" in content
        assert artifact is None


# ---------------------------------------------------------------------------
# render_image — success path & artifact shape
# ---------------------------------------------------------------------------


class TestRenderImageSuccess:
    async def test_png_accepted(self, tmp_path: Path) -> None:
        _make_png(tmp_path / "chart.png")
        tool = _get_tool(tmp_path)
        content, artifact = await _invoke(tool, "chart.png", caption="Monthly revenue")
        assert artifact is not None
        assert artifact["kind"] == IMAGE_ARTIFACT_KIND
        assert artifact["path"] == str((tmp_path / "chart.png").resolve())
        assert artifact["caption"] == "Monthly revenue"
        assert "Displayed image" in content

    async def test_jpeg_accepted(self, tmp_path: Path) -> None:
        path = tmp_path / "photo.jpg"
        Image.new("RGB", (4, 4), color="green").save(path, format="JPEG")
        tool = _get_tool(tmp_path)
        _, artifact = await _invoke(tool, "photo.jpg")
        assert artifact is not None

    async def test_bmp_accepted(self, tmp_path: Path) -> None:
        path = tmp_path / "diagram.bmp"
        Image.new("RGB", (4, 4), color="white").save(path, format="BMP")
        tool = _get_tool(tmp_path)
        _, artifact = await _invoke(tool, "diagram.bmp")
        assert artifact is not None

    async def test_webp_accepted(self, tmp_path: Path) -> None:
        path = tmp_path / "icon.webp"
        Image.new("RGB", (4, 4), color="black").save(path, format="WEBP")
        tool = _get_tool(tmp_path)
        _, artifact = await _invoke(tool, "icon.webp")
        assert artifact is not None

    async def test_tiff_accepted(self, tmp_path: Path) -> None:
        path = tmp_path / "scan.tiff"
        Image.new("RGB", (4, 4), color="yellow").save(path, format="TIFF")
        tool = _get_tool(tmp_path)
        _, artifact = await _invoke(tool, "scan.tiff")
        assert artifact is not None

    async def test_no_caption_defaults_to_empty_string(self, tmp_path: Path) -> None:
        _make_png(tmp_path / "chart.png")
        tool = _get_tool(tmp_path)
        _, artifact = await _invoke(tool, "chart.png")
        assert artifact["caption"] == ""

    async def test_nested_relative_path_resolved(self, tmp_path: Path) -> None:
        subdir = tmp_path / "outputs"
        subdir.mkdir()
        img = _make_png(subdir / "chart.png")
        tool = _get_tool(tmp_path)
        _, artifact = await _invoke(tool, "outputs/chart.png")
        assert artifact["path"] == str(img.resolve())


# ---------------------------------------------------------------------------
# render_image — image bytes never enter the LLM-visible content
# ---------------------------------------------------------------------------


class TestRenderImageDoesNotLeakBytes:
    async def test_content_string_never_contains_raw_image_bytes(self, tmp_path: Path) -> None:
        img_path = _make_png(tmp_path / "chart.png")
        tool = _get_tool(tmp_path)
        content, _ = await _invoke(tool, "chart.png")
        raw_bytes = img_path.read_bytes()
        assert raw_bytes not in content.encode("utf-8", errors="ignore")

    async def test_langchain_ainvoke_returns_only_content_string(self, tmp_path: Path) -> None:
        """The value the LLM actually sees (via .ainvoke) is a plain string,
        never the artifact dict — content_and_artifact keeps them separated."""
        _make_png(tmp_path / "chart.png")
        tool = _get_tool(tmp_path)
        result = await tool.ainvoke(
            {
                "artifact_path": "chart.png",
                "caption": "",
                "summary": "Displaying image",
                "communication": "",
            }
        )
        assert isinstance(result, str)
        assert "Displayed image" in result
