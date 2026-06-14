"""Color theme definitions for OpenDataSci TUI."""

# Default dark theme — deeper backgrounds, muted accents
DARK: dict[str, str] = {
    "accent": "#79c0ff",  # soft blue
    "success": "#4caa5e",  # muted sage-green
    "error": "#c97a74",  # muted terracotta-red
    "warning": "#c9963a",  # muted amber
    "thinking": "#9f75d6",  # muted lavender
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

# Color-blind safe theme — Okabe-Ito palette
VISIBLE: dict[str, str] = {
    "accent": "#56b4e9",  # sky blue
    "success": "#009e73",  # bluish green / teal
    "error": "#d55e00",  # vermilion
    "warning": "#e69f00",  # orange
    "thinking": "#cc79a7",  # reddish purple
    "text_primary": "#f5f5f5",
    "text_secondary": "#a0b0c0",
    "text_muted": "#8090a0",
    "text_dim": "#3d4a5c",
    "separator": "#1a2030",
    "logo": "#e69f00",
    "timer": "cyan",
    "tool_running": "#56b4e9",
    "tool_done": "#009e73",
}

# Light theme — light background, dark text (GitHub-inspired)
LIGHT: dict[str, str] = {
    "accent": "#0969da",
    "success": "#1a7f37",
    "error": "#cf222e",
    "warning": "#9a6700",
    "thinking": "#8250df",
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

# Solarized Dark — Ethan Schoonover palette
SOLARIZED: dict[str, str] = {
    "accent": "#268bd2",  # blue
    "success": "#859900",  # green
    "error": "#dc322f",  # red
    "warning": "#b58900",  # yellow
    "thinking": "#6c71c4",  # violet
    "text_primary": "#fdf6e3",  # base3
    "text_secondary": "#93a1a1",  # base1
    "text_muted": "#586e75",  # base01
    "text_dim": "#073642",  # base02
    "separator": "#002b36",  # base03
    "logo": "#cb4b16",  # orange
    "timer": "bright_cyan",
    "tool_running": "#268bd2",
    "tool_done": "#2aa198",  # cyan
}

# Dracula — popular dark palette
DRACULA: dict[str, str] = {
    "accent": "#bd93f9",  # purple
    "success": "#50fa7b",  # green
    "error": "#ff5555",  # red
    "warning": "#ffb86c",  # orange
    "thinking": "#ff79c6",  # pink
    "text_primary": "#f8f8f2",  # foreground
    "text_secondary": "#6272a4",  # comment
    "text_muted": "#44475a",
    "text_dim": "#282a36",
    "separator": "#21222c",
    "logo": "#f1fa8c",  # yellow
    "timer": "bright_magenta",
    "tool_running": "#8be9fd",  # cyan
    "tool_done": "#50fa7b",
}

# Registry of selectable themes. Keys are the names users pass to --theme
# and to the /themes command.
THEMES: dict[str, dict[str, str]] = {
    "default": DARK,
    "accessible": VISIBLE,
    "light": LIGHT,
    "solarized": SOLARIZED,
    "dracula": DRACULA,
}

THEME_DESCRIPTIONS: dict[str, str] = {
    "default": "Dark background with muted accents (built-in default)",
    "accessible": "Okabe-Ito palette — colour-blind safe",
    "light": "Light background with dark text",
    "solarized": "Solarized Dark by Ethan Schoonover",
    "dracula": "Dracula — vivid pastels on near-black",
}

# Mutated at startup by OpenDataSciApp based on --theme flag.
active: dict[str, str] = dict(DARK)
active_name: str = "default"
