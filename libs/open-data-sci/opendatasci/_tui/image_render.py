"""Image support for the chat pane: static-image format validation and the
browser-preview fallback for terminals with no native graphics protocol.

Inline pixel rendering is delegated to ``textual-image``'s ``AutoImage``
widget, which draws through the terminal's native graphics protocol (Kitty's
Terminal Graphics Protocol or Sixel); OpenDataSci deliberately never uses
that library's Unicode/half-cell fallback tier — its fidelity is too low to
be worth showing. Where no native protocol is available but a real
interactive terminal is, ``build_browser_preview`` backs a clickable OSC 8
hyperlink instead (see ``ImageBlock``) — nothing opens until the user clicks
it. See ``opendatasci._tui.graphics_utils`` for the terminal-capability
checks that choose between these.
"""

import html
import tempfile
from pathlib import Path

from PIL import Image, UnidentifiedImageError

# Static-image formats only. GIF is explicitly excluded (animated content has
# no single-frame meaning here); other multi-frame/video containers are
# rejected because Pillow simply won't recognize them as a still image.
_DISALLOWED_FORMATS = frozenset({"GIF"})

_PREVIEW_HTML_TEMPLATE = """\
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ background: #1e1e1e; color: #ddd; margin: 0; padding: 2rem;
          display: flex; flex-direction: column; align-items: center;
          font-family: system-ui, sans-serif; }}
  img {{ max-width: 90vw; max-height: 85vh; }}
  figcaption {{ margin-top: 1rem; text-align: center; }}
</style>
</head>
<body>
<figure style="margin: 0;">
  <img src="{image_uri}" alt="{alt}">
  {caption_html}
</figure>
</body>
</html>
"""


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


def build_browser_preview(path: Path, caption: str) -> str:
    """Write a standalone HTML page previewing *path* (with *caption*) and return its ``file://`` URI.

    The page references the image in place via its own ``file://`` URI
    rather than copying it, so it's only ever useful when the browser opening
    it can reach the same filesystem as this process — see ``ImageBlock``.
    """
    caption_html = f"<figcaption>{html.escape(caption)}</figcaption>" if caption else ""
    content = _PREVIEW_HTML_TEMPLATE.format(
        title=html.escape(path.name),
        image_uri=path.resolve().as_uri(),
        alt=html.escape(path.name),
        caption_html=caption_html,
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", prefix="opendatasci-image-", delete=False, encoding="utf-8"
    ) as fh:
        fh.write(content)
        preview_path = Path(fh.name)
    return preview_path.as_uri()
