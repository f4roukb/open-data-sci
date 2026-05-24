"""File-reference parsing for the @path/to/file syntax in user input."""

import re
import time
from pathlib import Path

from rich.markup import escape as escape_markup

# Matches @path/to/file  or  @Makefile  or  @.env
_FILE_REF_RE = re.compile(r"@(\S+?)(?=\s|$)")


class _FileRef:
    """A @file reference parsed from user input."""

    def __init__(self, path: str) -> None:
        self._path = path

    @property
    def display_name(self) -> str:
        return Path(self._path).name


def _parse_file_refs(text: str) -> tuple[str, list[_FileRef]]:
    """Return (text_without_refs, list_of_refs)."""
    refs = [_FileRef(path=m.group(1)) for m in _FILE_REF_RE.finditer(text)]
    clean = _FILE_REF_RE.sub("", text).strip()
    return clean, refs


def _build_user_display(clean_text: str, refs: list[_FileRef]) -> str:
    """Build the Rich-markup string shown in the user bubble."""
    parts = []
    for ref in refs:
        parts.append(rf"[bold #58a6ff]\[{ref.display_name}][/bold #58a6ff]")
    if clean_text:
        parts.append(escape_markup(clean_text))
    return "\n".join(parts)


def _build_agent_query(clean_text: str, refs: list[_FileRef]) -> str:
    """Build the query string sent to the agent, with file attachment tags."""
    if not refs:
        return clean_text
    parts = [clean_text] if clean_text else []
    for ref in refs:
        abs_path = str(Path(ref._path).resolve())
        parts.append(f'<file_attachment path="{abs_path}" name="{ref.display_name}"/>')
    return "\n\n".join(parts)


def _split_existing_file_refs(refs: list[_FileRef]) -> tuple[list[_FileRef], list[_FileRef]]:
    """Split refs into (existing_files, missing_or_invalid)."""
    existing: list[_FileRef] = []
    missing: list[_FileRef] = []
    for ref in refs:
        path = Path(ref._path).expanduser()
        if path.exists() and path.is_file():
            existing.append(ref)
        else:
            missing.append(ref)
    return existing, missing


def _find_slash_fragment(text: str) -> str | None:
    """Return the slash fragment if text starts with / and has no space yet."""
    if text.startswith("/") and " " not in text:
        return text
    return None


def _find_at_fragment(text: str) -> tuple[str, int] | None:
    """Return (fragment, at_index) for the last active @-reference being typed, or None."""
    at_pos = text.rfind("@")
    if at_pos == -1:
        return None
    after = text[at_pos + 1 :]
    space_pos = after.find(" ")
    fragment = after[:space_pos] if space_pos != -1 else after
    return fragment, at_pos


_DIR_CACHE_TTL = 2.0
# Maps str(search_dir) → (timestamp, [(name, is_dir), ...])
_dir_cache: dict[str, tuple[float, list[tuple[str, bool]]]] = {}


def _discover_files(fragment: str) -> list[str]:
    """Return up to 10 file/dir paths under cwd that match the typed fragment."""
    frag = fragment.replace("\\", "/")
    if "/" in frag:
        dir_part, name_prefix = frag.rsplit("/", 1)
        search_dir = Path(dir_part) if Path(dir_part).is_absolute() else Path.cwd() / dir_part
    else:
        dir_part = ""
        name_prefix = frag
        search_dir = Path.cwd()

    if not search_dir.is_dir():
        return []

    cache_key = str(search_dir)
    now = time.monotonic()
    cached = _dir_cache.get(cache_key)
    if cached is None or now - cached[0] > _DIR_CACHE_TTL:
        try:
            entries: list[tuple[str, bool]] = [
                (e.name, e.is_dir()) for e in sorted(search_dir.iterdir())
            ]
        except (PermissionError, OSError):
            return []
        _dir_cache[cache_key] = (now, entries)
    else:
        entries = cached[1]

    show_hidden = name_prefix.startswith(".")
    matches: list[str] = []
    for name, is_dir in entries:
        if name.startswith(".") and not show_hidden:
            continue
        if name_prefix and not name.lower().startswith(name_prefix.lower()):
            continue
        rel = (dir_part + "/" + name) if dir_part else name
        if is_dir:
            rel += "/"
        matches.append(rel)

    return matches[:10]


class PasteAttachment:
    """Multi-line text pasted by the user, shown as a compact pill in the UI.

    The LLM receives the full content inside ``<pasted_content>`` tags;
    the user sees only the pill label in the attachment bar.
    """

    def __init__(self, content: str) -> None:
        self._content = content
        self._line_count = content.count("\n") + 1

    @property
    def display_label(self) -> str:
        n = self._line_count
        return f"Text: {n} line{'s' if n != 1 else ''}"

    @property
    def pill_markup(self) -> str:
        return rf"[bold #58a6ff]\[{escape_markup(self.display_label)}][/bold #58a6ff]"

    @property
    def xml_tag(self) -> str:
        return f"<pasted_content>\n{self._content}\n</pasted_content>"
