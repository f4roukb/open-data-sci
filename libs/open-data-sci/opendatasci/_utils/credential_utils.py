import glob
import os
import sys


def _interpreter_startup_paths() -> list[str]:
    """Directories the sandboxed Python interpreter must be able to read to start
    up at all: its install prefix(es) and its own binary's directory. Covers
    venvs whose base interpreter (``sys.base_prefix``) lives somewhere other
    than the venv itself (``sys.prefix``), e.g. pyenv/uv-managed venvs."""
    return [sys.prefix, sys.base_prefix, os.path.dirname(sys.executable)]


def _is_within(parent: str, path: str) -> bool:
    """True if *path* is *parent* itself or lives somewhere underneath it."""
    return path == parent or path.startswith(parent + os.sep)


def _corridor_deny_paths(root: str, keep_paths: list[str]) -> list[str]:
    """Deny-read paths for everything under *root* except a minimal corridor
    down to each already-resolved path in *keep_paths*.

    Used when a directory that would otherwise be denied wholesale (e.g. a
    home dotdir matched by :func:`find_maybe_sensitive_paths`) turns out
    to be an ancestor of somewhere the sandbox must be able to read, such as
    the interpreter running the sandboxed code. Rather than exempting the
    whole directory — which could also expose unrelated sensitive data living
    next to it (e.g. ``~/.local/share/keyrings`` beside
    ``~/.local/share/uv``) — this walks from *root* down to each keep target
    and denies every sibling encountered along the way, so only the exact
    path to the keep target opens up.
    """
    relevant_keep_paths = [keep for keep in keep_paths if _is_within(root, keep)]
    deny_paths: list[str] = []

    def collect_denials(path: str) -> None:
        if any(_is_within(keep, path) for keep in relevant_keep_paths):
            return  # path is a keep target, or lives under one: fully readable
        if not any(_is_within(path, keep) for keep in relevant_keep_paths):
            deny_paths.append(path)  # unrelated to every keep target: deny outright
            return
        try:  # an ancestor of some keep target: recurse, denying its other children
            children = os.listdir(path)
        except OSError:
            return
        for child in children:
            collect_denials(os.path.join(path, child))

    collect_denials(root)
    return deny_paths


def find_maybe_sensitive_paths() -> list[str]:
    """Host locations that might be sensitive, identified heuristically by
    naming convention — not a confirmed or exhaustive list of credential
    stores.

    Denies every top-level dotfile/dotdir directly under the home directory
    by naming convention — ``~/.ssh``, ``~/.aws``, ``~/.netrc``,
    ``~/.config`` (and everything under it), etc. — so it also catches
    credential stores we haven't explicitly enumerated. Only reaches one
    level deep (``~/.foo``, not ``~/projects/.env``). Also covers a few
    macOS locations that don't follow the dotfile convention: Keychain,
    and WebKit/browser cookie jars under ``~/Library``.

    If one of those directories turns out to be an ancestor of the Python
    interpreter running the sandbox itself (pyenv/uv/rye commonly install
    interpreters under a home dotdir, e.g. ``~/.local`` or ``~/.pyenv``), it
    is not exempted outright — that would also reopen unrelated sensitive
    data that might live alongside the toolchain. Instead only a minimal
    read corridor down to the interpreter is carved out via
    :func:`_corridor_deny_paths`, keeping everything else in that directory
    denied.
    """
    home = os.path.realpath(os.path.expanduser("~"))
    candidate_paths: list[str] = []
    candidate_paths.extend(glob.glob(os.path.join(home, ".*")))

    # macOS Keychain: the platform's primary credential store. Not a dotfile
    # (lives under ``~/Library``), so the glob above never reaches it.
    candidate_paths.append(os.path.join(home, "Library", "Keychains"))

    # macOS browser/WebKit cookie jars, which commonly hold live session
    # tokens and likewise live outside ``~/Library``'s non-dot prefix.
    candidate_paths.append(os.path.join(home, "Library", "Cookies"))
    candidate_paths.append(os.path.join(home, "Library", "HTTPStorages"))

    interpreter_paths = [os.path.realpath(path) for path in _interpreter_startup_paths()]

    deny_paths: list[str] = []
    for candidate in candidate_paths:
        if not os.path.exists(candidate):
            continue
        resolved = os.path.realpath(candidate)
        if any(_is_within(resolved, keep) for keep in interpreter_paths):
            deny_paths.extend(_corridor_deny_paths(resolved, interpreter_paths))
        else:
            deny_paths.append(resolved)

    return deny_paths
