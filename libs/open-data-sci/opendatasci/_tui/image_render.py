"""Image support for the chat pane: static-image format validation.

Actual rendering is delegated to ``textual-image``'s ``AutoImage`` widget,
which draws through the terminal's native graphics protocol (Kitty's
Terminal Graphics Protocol or Sixel). OpenDataSci deliberately never uses
that library's Unicode/half-cell fallback tier — its fidelity is too low to
be worth showing — so ``render_image`` is only ever offered to the agent
(``OpenDataSciConfig.enable_image_rendering``) when a real graphics protocol
was detected; see ``opendatasci._utils.graphics_utils.terminal_supports_image_graphics``.
"""

from pathlib import Path

from PIL import Image, UnidentifiedImageError

# Static-image formats only. GIF is explicitly excluded (animated content has
# no single-frame meaning here); other multi-frame/video containers are
# rejected because Pillow simply won't recognize them as a still image.
_DISALLOWED_FORMATS = frozenset({"GIF"})


class UnsupportedImageError(ValueError):
    """Raised when a file is missing, unreadable, or not a supported static image."""


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
