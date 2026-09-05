"""Runtime toggle for the footer tips bar (see /config > Display > Tips).

Session-local, like the active theme (``style/theme.py``) — resets to
enabled on every launch; ``TipsBar`` reads it on each render.
"""

enabled: bool = True


def set_enabled(value: bool) -> None:
    global enabled
    enabled = value
