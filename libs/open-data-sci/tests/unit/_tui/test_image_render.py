"""Unit tests for opendatasci._tui.image_render."""

from pathlib import Path

import pytest
from PIL import Image

from opendatasci._tui.image_render import (
    UnsupportedImageError,
    terminal_supports_image_graphics,
    validate_static_image,
)

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


# ---------------------------------------------------------------------------
# terminal_supports_image_graphics
# ---------------------------------------------------------------------------


class TestTerminalSupportsImageGraphics:
    def test_true_when_sixel_detected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from textual_image.renderable.sixel import Image as SixelRenderable

        monkeypatch.setattr(
            "opendatasci._tui.image_render._DetectedImageRenderable", SixelRenderable
        )
        assert terminal_supports_image_graphics() is True

    def test_true_when_tgp_detected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from textual_image.renderable.tgp import Image as TGPRenderable

        monkeypatch.setattr("opendatasci._tui.image_render._DetectedImageRenderable", TGPRenderable)
        assert terminal_supports_image_graphics() is True

    def test_false_when_halfcell_detected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from textual_image.renderable.halfcell import Image as HalfcellRenderable

        monkeypatch.setattr(
            "opendatasci._tui.image_render._DetectedImageRenderable", HalfcellRenderable
        )
        assert terminal_supports_image_graphics() is False

    def test_false_when_unicode_detected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from textual_image.renderable.unicode import Image as UnicodeRenderable

        monkeypatch.setattr(
            "opendatasci._tui.image_render._DetectedImageRenderable", UnicodeRenderable
        )
        assert terminal_supports_image_graphics() is False

    def test_reflects_actual_test_environment_detection(self) -> None:
        """Under pytest stdout isn't a real tty, so textual-image resolves to its
        Unicode fallback at import time — this must never read as graphics support."""
        assert terminal_supports_image_graphics() is False
