"""Image support for the chat pane: format validation and terminal capability detection.

Actual rendering is delegated to ``textual-image``'s ``AutoImage`` widget,
which draws through the terminal's native graphics protocol (Kitty's
Terminal Graphics Protocol or Sixel). OpenDataSci deliberately never uses
that library's Unicode/half-cell fallback tier — its fidelity is too low to
be worth showing — so ``render_image`` is only ever offered to the agent
(``OpenDataSciConfig.enable_image_rendering``) when a real graphics protocol
was detected; see ``terminal_supports_image_graphics``.
"""

from pathlib import Path

from PIL import Image, UnidentifiedImageError
from textual_image.renderable import Image as _DetectedImageRenderable
from textual_image.renderable.sixel import Image as _SixelRenderable
from textual_image.renderable.tgp import Image as _TGPRenderable

# Static-image formats only. GIF is explicitly excluded (animated content has
# no single-frame meaning here); other multi-frame/video containers are
# rejected because Pillow simply won't recognize them as a still image.
_DISALLOWED_FORMATS = frozenset({"GIF"})


class UnsupportedImageError(ValueError):
    """Raised when a file is missing, unreadable, or not a supported static image."""


def terminal_supports_image_graphics() -> bool:
    """True only when the terminal supports a real graphics protocol (Kitty TGP or Sixel).

    False for every other case, including the half-cell/Unicode fallback
    tiers — those are excluded on purpose, never used for rendering here.
    """
    return _DetectedImageRenderable in (_SixelRenderable, _TGPRenderable)


def validate_static_image(path: Path) -> None:
    """Raise ``UnsupportedImageError`` unless *path* is a real, static (non-GIF) image.

    Performs a structural check only (Pillow's ``verify()``) — never decodes
    or returns pixel data; the caller (``ImageBlock``) hands the path
    straight to ``AutoImage`` for actual rendering once this passes.
    """
    try:
        with Image.open(path) as img:
            image_format = img.format
            img.verify()
    except FileNotFoundError as exc:
        raise UnsupportedImageError(f"Image not found: {path}") from exc
    except UnidentifiedImageError as exc:
        raise UnsupportedImageError(f"'{path.name}' is not a recognized image file.") from exc
    except OSError as exc:
        raise UnsupportedImageError(f"Could not read '{path.name}': {exc}") from exc
    if image_format in _DISALLOWED_FORMATS:
        raise UnsupportedImageError(
            f"'{path.name}' is a {image_format} — only static images are supported."
        )
