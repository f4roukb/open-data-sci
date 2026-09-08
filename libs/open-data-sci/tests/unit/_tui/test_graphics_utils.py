"""Unit tests for opendatasci._tui.graphics_utils."""

import sys

import pytest

from opendatasci._tui.graphics_utils import (
    terminal_is_interactive,
    terminal_supports_image_graphics,
)

# ---------------------------------------------------------------------------
# terminal_supports_image_graphics
# ---------------------------------------------------------------------------


class TestTerminalSupportsImageGraphics:
    def test_true_when_sixel_detected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from textual_image.renderable.sixel import Image as SixelRenderable

        monkeypatch.setattr(
            "opendatasci._tui.graphics_utils._DetectedImageRenderable", SixelRenderable
        )
        assert terminal_supports_image_graphics() is True

    def test_true_when_tgp_detected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from textual_image.renderable.tgp import Image as TGPRenderable

        monkeypatch.setattr(
            "opendatasci._tui.graphics_utils._DetectedImageRenderable", TGPRenderable
        )
        assert terminal_supports_image_graphics() is True

    def test_false_when_halfcell_detected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from textual_image.renderable.halfcell import Image as HalfcellRenderable

        monkeypatch.setattr(
            "opendatasci._tui.graphics_utils._DetectedImageRenderable", HalfcellRenderable
        )
        assert terminal_supports_image_graphics() is False

    def test_false_when_unicode_detected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from textual_image.renderable.unicode import Image as UnicodeRenderable

        monkeypatch.setattr(
            "opendatasci._tui.graphics_utils._DetectedImageRenderable", UnicodeRenderable
        )
        assert terminal_supports_image_graphics() is False

    def test_reflects_actual_test_environment_detection(self) -> None:
        """Under pytest stdout isn't a real tty, so textual-image resolves to its
        Unicode fallback at import time — this must never read as graphics support."""
        assert terminal_supports_image_graphics() is False


# ---------------------------------------------------------------------------
# terminal_is_interactive
# ---------------------------------------------------------------------------


class TestTerminalIsInteractive:
    def test_true_when_stdout_is_a_tty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _FakeTTYStdout:
            def isatty(self) -> bool:
                return True

        monkeypatch.setattr(sys, "__stdout__", _FakeTTYStdout())
        assert terminal_is_interactive() is True

    def test_false_when_stdout_is_not_a_tty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _FakeNonTTYStdout:
            def isatty(self) -> bool:
                return False

        monkeypatch.setattr(sys, "__stdout__", _FakeNonTTYStdout())
        assert terminal_is_interactive() is False

    def test_false_when_stdout_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "__stdout__", None)
        assert terminal_is_interactive() is False

    def test_reflects_actual_test_environment_detection(self) -> None:
        """Under pytest stdout isn't a real tty."""
        assert terminal_is_interactive() is False
