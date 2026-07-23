"""Unit tests for opendatasci._tui.theme.

No prior test file pinned the palette/REQUIRED_KEYS contract, which is how a
theme key (``thinking``, ``muted_bg``) survived in every palette for months
after its only consumer (a dead ``MessageBubble.thinking`` CSS rule) was
removed. These tests guard the contract going forward.
"""

from opendatasci._tui import theme as _theme

# NOTE: the autouse `_restore_active_theme` fixture in conftest.py resets
# `_theme.active`/`active_name` after every test in this package, so tests
# below are free to call set_active() without cleaning up manually.


class TestRequiredKeys:
    def test_required_keys_matches_dark_palette_keys(self) -> None:
        assert _theme.REQUIRED_KEYS == frozenset(_theme.DARK)

    def test_every_registered_theme_defines_exactly_required_keys(self) -> None:
        for name, palette in _theme.THEMES.items():
            assert set(palette.keys()) == _theme.REQUIRED_KEYS, (
                f"theme {name!r} keys diverge from REQUIRED_KEYS: "
                f"missing={_theme.REQUIRED_KEYS - palette.keys()}, "
                f"extra={palette.keys() - _theme.REQUIRED_KEYS}"
            )


class TestThemeRegistry:
    def test_default_theme_is_dark(self) -> None:
        assert _theme.THEMES["default"] is _theme.DARK

    def test_every_theme_has_a_description(self) -> None:
        assert set(_theme.THEME_DESCRIPTIONS.keys()) == set(_theme.THEMES.keys())

    def test_active_defaults_to_dark_values(self) -> None:
        assert _theme.active == _theme.DARK

    def test_active_name_defaults_to_default(self) -> None:
        assert _theme.active_name == "default"


class TestSetActive:
    """set_active() powers live in-TUI theme switching via /theme <name>."""

    def test_valid_name_returns_true(self) -> None:
        assert _theme.set_active("dracula") is True

    def test_valid_name_updates_active_palette(self) -> None:
        _theme.set_active("light")
        assert _theme.active == _theme.LIGHT

    def test_valid_name_updates_active_name(self) -> None:
        _theme.set_active("solarized")
        assert _theme.active_name == "solarized"

    def test_unknown_name_returns_false(self) -> None:
        assert _theme.set_active("does-not-exist") is False

    def test_unknown_name_leaves_active_palette_unchanged(self) -> None:
        _theme.set_active("accessible")
        before = dict(_theme.active)
        assert _theme.set_active("nope") is False
        assert _theme.active == before

    def test_unknown_name_leaves_active_name_unchanged(self) -> None:
        _theme.set_active("dracula")
        assert _theme.set_active("nope") is False
        assert _theme.active_name == "dracula"

    def test_switching_back_to_default_restores_dark(self) -> None:
        _theme.set_active("light")
        _theme.set_active("default")
        assert _theme.active == _theme.DARK
        assert _theme.active_name == "default"
