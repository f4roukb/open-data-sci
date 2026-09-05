"""Render a static image file as truecolor ANSI art for the chat pane.

No terminal graphics protocol (Sixel/Kitty/iTerm2) is assumed — half-block
Unicode characters with per-cell truecolor foreground/background styling
render correctly in any modern terminal Textual already targets, and need no
extra runtime dependency beyond Pillow (used purely to decode pixels; any
format Pillow can open is accepted, GIF excluded).
"""

from pathlib import Path

from PIL import Image, UnidentifiedImageError
from rich.text import Text

DEFAULT_MAX_WIDTH = 60
_UPPER_HALF_BLOCK = "▀"
# See RenderImageTool._DISALLOWED_FORMATS — kept in sync: GIF (animated) and
# any other non-still format are rejected at both the tool and render layers.
_DISALLOWED_FORMATS = frozenset({"GIF"})


class UnsupportedImageError(ValueError):
    """Raised when a file is missing, unreadable, or not a supported static image."""


def render_image_to_text(path: Path, max_width: int = DEFAULT_MAX_WIDTH) -> Text:
    """Decode the image at *path* and return it as a Rich ``Text`` of half-block art.

    Each output character cell encodes two source pixel rows (top pixel as
    foreground, bottom pixel as background of an upper-half-block glyph), and
    the image is downscaled to fit *max_width* columns. Raises
    ``UnsupportedImageError`` for a missing file, an unreadable/corrupt file,
    or a disallowed format (GIF).
    """
    try:
        with Image.open(path) as img:
            if img.format in _DISALLOWED_FORMATS:
                raise UnsupportedImageError(
                    f"'{path.name}' is a {img.format} — only static images are supported."
                )
            rgb = img.convert("RGB")
    except FileNotFoundError as exc:
        raise UnsupportedImageError(f"Image not found: {path}") from exc
    except UnidentifiedImageError as exc:
        raise UnsupportedImageError(f"'{path.name}' is not a recognized image file.") from exc
    except OSError as exc:
        raise UnsupportedImageError(f"Could not read '{path.name}': {exc}") from exc

    rgb = _fit_to_width(rgb, max_width)
    width, height = rgb.size
    if height % 2:
        # Odd pixel height: duplicate the last row so every pair has a partner.
        padded = Image.new("RGB", (width, height + 1))
        padded.paste(rgb, (0, 0))
        padded.paste(rgb.crop((0, height - 1, width, height)), (0, height))
        rgb = padded
        height += 1

    text = Text()
    for y in range(0, height, 2):
        for x in range(width):
            top_r, top_g, top_b = _pixel(rgb, x, y)
            bot_r, bot_g, bot_b = _pixel(rgb, x, y + 1)
            style = f"rgb({top_r},{top_g},{top_b}) on rgb({bot_r},{bot_g},{bot_b})"
            text.append(_UPPER_HALF_BLOCK, style=style)
        if y + 2 < height:
            text.append("\n")
    return text


def _pixel(img: Image.Image, x: int, y: int) -> tuple[int, int, int]:
    """Return the RGB triple at (x, y) — img is always in "RGB" mode here."""
    r, g, b = img.getpixel((x, y))  # type: ignore[misc]
    return r, g, b


def _fit_to_width(img: Image.Image, max_width: int) -> Image.Image:
    """Downscale *img* to at most *max_width* columns, preserving aspect ratio.

    Terminal character cells are roughly twice as tall as wide, so the target
    pixel height is halved relative to a naive aspect-preserving scale to keep
    the rendered image visually proportionate.
    """
    width, height = img.size
    if width <= max_width:
        return img
    scale = max_width / width
    new_width = max_width
    new_height = max(1, round(height * scale * 0.5))
    return img.resize((new_width, new_height))
