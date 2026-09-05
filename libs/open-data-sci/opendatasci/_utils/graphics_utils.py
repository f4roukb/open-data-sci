"""Terminal graphics-capability detection.

Pure and dependency-free of any particular caller -- this only asks what the
terminal itself supports; it says nothing about how any subsystem chooses to
use that answer. ``textual-image`` resolves this once, at import time, by
querying the real terminal, so the check here is a cheap lookup rather than
a fresh probe.
"""

from textual_image.renderable import Image as _DetectedImageRenderable
from textual_image.renderable.sixel import Image as _SixelRenderable
from textual_image.renderable.tgp import Image as _TGPRenderable


def terminal_supports_image_graphics() -> bool:
    """True only when the terminal supports a real graphics protocol (Kitty TGP or Sixel).

    False for every other case, including ``textual-image``'s half-cell/Unicode
    fallback tiers -- callers that care about rendering fidelity should treat
    those the same as no support at all.
    """
    return _DetectedImageRenderable in (_SixelRenderable, _TGPRenderable)
