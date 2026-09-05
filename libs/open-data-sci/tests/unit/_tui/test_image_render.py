"""Unit tests for opendatasci._tui.image_render."""

from pathlib import Path

import pytest
from PIL import Image

from opendatasci._tui.image_render import (
    DEFAULT_MAX_WIDTH,
    UnsupportedImageError,
    render_image_to_text,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save(path: Path, size: tuple[int, int], color, fmt: str = "PNG") -> Path:
    Image.new("RGB", size, color=color).save(path, format=fmt)
    return path


# ---------------------------------------------------------------------------
# Missing / invalid files
# ---------------------------------------------------------------------------


class TestRenderImageToTextErrors:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(UnsupportedImageError, match="not found"):
            render_image_to_text(tmp_path / "missing.png")

    def test_non_image_file_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "notes.txt"
        bad.write_text("hello")
        with pytest.raises(UnsupportedImageError, match="not a recognized image"):
            render_image_to_text(bad)

    def test_corrupt_bytes_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "chart.png"
        bad.write_bytes(b"not actually a png")
        with pytest.raises(UnsupportedImageError):
            render_image_to_text(bad)

    def test_gif_raises(self, tmp_path: Path) -> None:
        gif = _save(tmp_path / "anim.gif", (2, 2), "blue", fmt="GIF")
        with pytest.raises(UnsupportedImageError, match="GIF"):
            render_image_to_text(gif)


# ---------------------------------------------------------------------------
# Pixel-to-ANSI mapping
# ---------------------------------------------------------------------------


class TestRenderImageToTextMapping:
    def test_two_by_two_solid_colors_maps_top_and_bottom_rows(self, tmp_path: Path) -> None:
        img = Image.new("RGB", (2, 2))
        img.putpixel((0, 0), (255, 0, 0))
        img.putpixel((1, 0), (255, 0, 0))
        img.putpixel((0, 1), (0, 0, 255))
        img.putpixel((1, 1), (0, 0, 255))
        path = tmp_path / "two_rows.png"
        img.save(path, format="PNG")

        text = render_image_to_text(path, max_width=DEFAULT_MAX_WIDTH)

        assert text.plain == "▀▀"
        spans = text.spans
        assert len(spans) == 2
        for span in spans:
            assert "rgb(255,0,0)" in span.style
            assert "rgb(0,0,255)" in span.style

    def test_single_pixel_row_is_padded_to_a_full_cell(self, tmp_path: Path) -> None:
        img = Image.new("RGB", (1, 1), color=(10, 20, 30))
        path = tmp_path / "one_row.png"
        img.save(path, format="PNG")

        text = render_image_to_text(path)

        assert text.plain == "▀"

    def test_multi_row_output_joined_with_newlines(self, tmp_path: Path) -> None:
        img = Image.new("RGB", (1, 4), color=(1, 2, 3))
        path = tmp_path / "four_rows.png"
        img.save(path, format="PNG")

        text = render_image_to_text(path)

        # 4 pixel rows -> 2 terminal rows -> one newline between them.
        assert text.plain == "▀\n▀"

    def test_wide_image_is_downscaled_to_max_width(self, tmp_path: Path) -> None:
        img = Image.new("RGB", (200, 100), color="green")
        path = tmp_path / "wide.png"
        img.save(path, format="PNG")

        text = render_image_to_text(path, max_width=40)

        # Every row of the rendered text must be at most 40 glyphs wide.
        for line in text.plain.split("\n"):
            assert len(line) <= 40

    def test_narrow_image_is_not_upscaled(self, tmp_path: Path) -> None:
        img = Image.new("RGB", (3, 2), color="purple")
        path = tmp_path / "narrow.png"
        img.save(path, format="PNG")

        text = render_image_to_text(path, max_width=DEFAULT_MAX_WIDTH)

        assert len(text.plain.split("\n")[0]) == 3


# ---------------------------------------------------------------------------
# Format coverage — anything Pillow can decode except GIF
# ---------------------------------------------------------------------------


class TestRenderImageToTextFormats:
    @pytest.mark.parametrize("fmt,ext", [("JPEG", "jpg"), ("BMP", "bmp"), ("WEBP", "webp")])
    def test_static_formats_render_without_error(
        self, tmp_path: Path, fmt: str, ext: str
    ) -> None:
        path = _save(tmp_path / f"image.{ext}", (4, 4), "orange", fmt=fmt)
        text = render_image_to_text(path)
        assert text.plain
