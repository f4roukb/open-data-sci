import glob
import os


def discover_credential_deny_paths() -> list[str]:
    """Paths to deny IO operations on common credential locations. Not comprehensive."""
    home = os.path.expanduser("~")
    candidates: list[str] = []

    # Every top-level dotfile/dotdir directly under the home directory:
    # ``~/.ssh``, ``~/.aws``, ``~/.netrc``, ``~/.config`` (and everything
    # under it), etc. Blanket-denies by naming convention rather than an
    # explicit allowlist of known tools, so it also catches ones we haven't
    # enumerated. Only reaches one level deep (``~/.foo``, not
    # ``~/projects/.env``).
    candidates.extend(glob.glob(os.path.join(home, ".*")))

    # macOS Keychain: the platform's primary credential store. Not a dotfile
    # (lives under ``~/Library``), so the glob above never reaches it.
    candidates.append(os.path.join(home, "Library", "Keychains"))

    # macOS browser/WebKit cookie jars, which commonly hold live session
    # tokens and likewise live outside ``~/Library``'s non-dot prefix.
    candidates.append(os.path.join(home, "Library", "Cookies"))
    candidates.append(os.path.join(home, "Library", "HTTPStorages"))

    return [os.path.realpath(path) for path in candidates if os.path.exists(path)]
