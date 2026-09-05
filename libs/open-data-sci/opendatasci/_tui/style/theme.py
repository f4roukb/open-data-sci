"""Color theme definitions for OpenDataSci TUI.

Every theme must define the full set of keys (see ``REQUIRED_KEYS``): the
palettes are exported as Textual CSS variables via
``OpenDataSciApp.get_css_variables`` and referenced from ``styles.tcss`` and
widget ``DEFAULT_CSS`` — no color literal should appear outside this module.
"""

# Default dark theme — deeper backgrounds, muted blue accents
DARK: dict[str, str] = {
    "background": "#060a10",  # screen background
    "surface": "#0d1117",  # header / input / popup / code-fence background
    "surface_alt": "#0e1520",  # user bubble / blockquote / scrollbar track
    "warning_bg": "#1c1408",  # question / pending bubble background
    "accent": "#79c0ff",  # soft blue
    "success": "#4caa5e",  # muted sage-green
    "error": "#c97a74",  # muted terracotta-red
    "warning": "#c9963a",  # muted amber
    "text_primary": "#e6edf3",
    "text_secondary": "#8b949e",
    "text_muted": "#6e7681",
    "text_dim": "#21262d",
    "separator": "#1a2030",
    "logo": "#c9a86c",
    "timer": "bright_green",
    "tool_running": "#58a6ff",
    "tool_done": "#3fb950",
}

# Light theme — light background, dark text (GitHub-inspired)
LIGHT: dict[str, str] = {
    "background": "#ffffff",
    "surface": "#f6f8fa",
    "surface_alt": "#eaeef2",
    "warning_bg": "#fff8c5",
    "accent": "#0969da",
    "success": "#1a7f37",
    "error": "#cf222e",
    "warning": "#9a6700",
    "text_primary": "#1f2328",
    "text_secondary": "#656d76",
    "text_muted": "#8c959f",
    "text_dim": "#d0d7de",
    "separator": "#d8dee4",
    "logo": "#bf8700",
    "timer": "blue",
    "tool_running": "#0969da",
    "tool_done": "#1a7f37",
}

# Dark, colour-blind safe — Okabe-Ito palette on a dark background
DARK_COLORBLIND: dict[str, str] = {
    "background": "#060a10",
    "surface": "#0d1117",
    "surface_alt": "#0a1520",
    "warning_bg": "#1c1608",
    "accent": "#56b4e9",  # sky blue
    "success": "#009e73",  # bluish green / teal
    "error": "#d55e00",  # vermilion
    "warning": "#e69f00",  # orange
    "text_primary": "#f5f5f5",
    "text_secondary": "#a0b0c0",
    "text_muted": "#8090a0",
    "text_dim": "#3d4a5c",
    "separator": "#1e2d40",
    "logo": "#e69f00",
    "timer": "cyan",
    "tool_running": "#56b4e9",
    "tool_done": "#009e73",
}

# Light, colour-blind safe — same Okabe-Ito hues as DARK_COLORBLIND, darkened
# where needed for contrast against a white background (mirrors how LIGHT
# darkens DARK's colors).
LIGHT_COLORBLIND: dict[str, str] = {
    "background": "#ffffff",
    "surface": "#f6f8fa",
    "surface_alt": "#eaeef2",
    "warning_bg": "#fff3e0",
    "accent": "#0072b2",  # Okabe-Ito blue, darkened for contrast on white
    "success": "#009e73",  # bluish green / teal
    "error": "#d55e00",  # vermilion
    "warning": "#9a5b00",  # Okabe-Ito orange, darkened for contrast on white
    "text_primary": "#1f2328",
    "text_secondary": "#656d76",
    "text_muted": "#8c959f",
    "text_dim": "#d0d7de",
    "separator": "#d8dee4",
    "logo": "#9a5b00",
    "timer": "blue",
    "tool_running": "#0072b2",
    "tool_done": "#009e73",
}

# The keys every palette must define — styles.tcss and widget DEFAULT_CSS
# reference each of these as a $ods-* CSS variable.
REQUIRED_KEYS: frozenset[str] = frozenset(DARK)

# Registry of selectable themes, in display order. Keys are never typed by
# the user — they're picked from a list in /config ▸ Display ▸ Theme — so
# they double as the display label.
THEMES: dict[str, dict[str, str]] = {
    "dark, colorblind": DARK_COLORBLIND,
    "dark": DARK,
    "light": LIGHT,
    "light (colorblind)": LIGHT_COLORBLIND,
}

THEME_DESCRIPTIONS: dict[str, str] = {
    "dark, colorblind": "Dark background, Okabe-Ito colour-blind safe palette",
    "dark": "Dark background with muted blue accents",
    "light": "Light background with dark text",
    "light (colorblind)": "Light background, Okabe-Ito colour-blind safe palette",
}

# Mutated at runtime by set_active() when the user switches themes via
# /config ▸ Display ▸ Theme (and once, for the initial pick, by the
# mandatory startup wizard).
active: dict[str, str] = dict(DARK_COLORBLIND)
active_name: str = "dark, colorblind"


def set_active(name: str) -> bool:
    """Switch the active palette to *name* in place.

    Returns ``True`` and updates ``active``/``active_name`` when *name* is a
    registered theme; returns ``False`` and leaves the current theme
    untouched otherwise, so callers can tell a valid switch from a no-op.
    """
    global active_name
    if name not in THEMES:
        return False
    active.clear()
    active.update(THEMES[name])
    active_name = name
    return True
