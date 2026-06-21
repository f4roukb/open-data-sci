import asyncio
from pathlib import Path
from typing import cast

import xxhash

from opendatasci._utils.async_utils import run_in_executor

_HASHING_DATA_CHUNK_SIZE = 32 * 1024 * 1024  # 32MB


def _is_visible(file: Path, root: Path) -> bool:
    """Return ``True`` if *file* is not under a hidden (dot-prefixed) path."""
    return not any(part.startswith(".") for part in file.relative_to(root).parts)


async def hash_path(path: Path) -> str:
    """Return the 128-bit xxh3 hex digest of *path* (file or directory)."""
    if path.is_file():
        return cast(str, await run_in_executor(hash_file, path))
    return await hash_dir(path)


def hash_file(path: Path, chunk_size: int | None = None) -> str:
    """Return the 128-bit xxh3 hex digest of a file's contents."""
    chunk_size = chunk_size or _HASHING_DATA_CHUNK_SIZE
    h = xxhash.xxh3_128()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


async def hash_dir(path: Path) -> str:
    """Return a 128-bit hash of a directory tree, excluding hidden/system files."""
    files: list[Path] = sorted(
        (f for f in path.rglob("*") if f.is_file() and _is_visible(f, path)),
        key=lambda p: str(p),
    )

    if not files:
        return xxhash.xxh3_128(b"").hexdigest()

    coros = [run_in_executor(hash_file, f) for f in files]
    hashes: list[str] = await asyncio.gather(*coros)

    h_prev: bytes = b""
    for h in hashes:
        file_hash = xxhash.xxh3_128(bytes.fromhex(h)).digest()
        h_prev = xxhash.xxh3_128(h_prev + file_hash).digest()

    return h_prev.hex()
