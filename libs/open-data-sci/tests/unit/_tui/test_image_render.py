"""Unit tests for opendatasci._tui.image_render."""

from pathlib import Path

import pytest
from PIL import Image

from opendatasci._tui.image_render import UnsupportedImageError, validate_static_image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save(path: Path, fmt: str = "PNG") -> Path:
    Image.new("RGB", (4, 4), color="red").save(path, format=fmt)
    return path


# ---------------------------------------------------------------------------
# validate_static_image
# ---------------------------------------------------------------------------


class TestValidateStaticImage:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(UnsupportedImageError, match="not found"):
            validate_static_image(tmp_path / "missing.png")

    def test_non_image_file_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "notes.txt"
        bad.write_text("hello")
        with pytest.raises(UnsupportedImageError, match="not a recognized image"):
            validate_static_image(bad)

    def test_corrupt_bytes_raise(self, tmp_path: Path) -> None:
        bad = tmp_path / "chart.png"
        bad.write_bytes(b"not actually a png")
        with pytest.raises(UnsupportedImageError):
            validate_static_image(bad)

    def test_gif_raises(self, tmp_path: Path) -> None:
        gif = _save(tmp_path / "anim.gif", fmt="GIF")
        with pytest.raises(UnsupportedImageError, match="GIF"):
            validate_static_image(gif)

    @pytest.mark.parametrize("fmt", ["PNG", "JPEG", "BMP", "WEBP", "TIFF"])
    def test_static_formats_pass(self, tmp_path: Path, fmt: str) -> None:
        path = _save(tmp_path / f"image.{fmt.lower()}", fmt=fmt)
        validate_static_image(path)  # must not raise
