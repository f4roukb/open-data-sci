"""Unit tests for opendatasci._utils.hash_utils."""


from pathlib import Path

import pytest
import xxhash

from opendatasci._utils.hash_utils import hash_dir, hash_file, hash_path


class TestHashFile:
    def test_returns_32_char_hex_string(self, tmp_path: Path) -> None:
        f = tmp_path / "data.bin"
        f.write_bytes(b"hello")
        result = hash_file(f)
        assert len(result) == 32
        assert all(c in "0123456789abcdef" for c in result)

    def test_same_content_same_hash(self, tmp_path: Path) -> None:
        a = tmp_path / "a.bin"
        b = tmp_path / "b.bin"
        a.write_bytes(b"same content")
        b.write_bytes(b"same content")
        assert hash_file(a) == hash_file(b)

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        a = tmp_path / "a.bin"
        b = tmp_path / "b.bin"
        a.write_bytes(b"content A")
        b.write_bytes(b"content B")
        assert hash_file(a) != hash_file(b)

    def test_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")
        result = hash_file(f)
        assert result == xxhash.xxh3_128(b"").hexdigest()

    def test_multi_chunk_matches_single_chunk(self, tmp_path: Path) -> None:
        content = b"x" * (3 * 1024 * 1024)  # 3 MB, spans multiple chunks at 1 MB boundary
        f = tmp_path / "large.bin"
        f.write_bytes(content)
        assert hash_file(f, chunk_size=1024 * 1024) == hash_file(f, chunk_size=len(content))

    def test_custom_chunk_size_produces_same_hash(self, tmp_path: Path) -> None:
        content = b"abcdefgh" * 1000
        f = tmp_path / "data.bin"
        f.write_bytes(content)
        assert hash_file(f, chunk_size=128) == hash_file(f, chunk_size=4096)

    def test_hash_changes_when_file_content_changes(self, tmp_path: Path) -> None:
        f = tmp_path / "data.bin"
        f.write_bytes(b"version 1")
        h1 = hash_file(f)
        f.write_bytes(b"version 2")
        h2 = hash_file(f)
        assert h1 != h2


class TestHashDir:
    @pytest.mark.asyncio
    async def test_empty_directory_returns_hex_string(self, tmp_path: Path) -> None:
        result = await hash_dir(tmp_path)
        assert result == xxhash.xxh3_128(b"").hexdigest()

    @pytest.mark.asyncio
    async def test_same_files_same_hash(self, tmp_path: Path) -> None:
        d1 = tmp_path / "d1"
        d2 = tmp_path / "d2"
        d1.mkdir()
        d2.mkdir()
        (d1 / "a.txt").write_bytes(b"hello")
        (d2 / "a.txt").write_bytes(b"hello")
        assert await hash_dir(d1) == await hash_dir(d2)

    @pytest.mark.asyncio
    async def test_different_file_content_different_hash(self, tmp_path: Path) -> None:
        d1 = tmp_path / "d1"
        d2 = tmp_path / "d2"
        d1.mkdir()
        d2.mkdir()
        (d1 / "a.txt").write_bytes(b"hello")
        (d2 / "a.txt").write_bytes(b"world")
        assert await hash_dir(d1) != await hash_dir(d2)

    @pytest.mark.asyncio
    async def test_excludes_hidden_files(self, tmp_path: Path) -> None:
        d = tmp_path / "dataset"
        d.mkdir()
        (d / "data.csv").write_bytes(b"a,b\n1,2\n")
        h_before = await hash_dir(d)
        (d / ".hidden").write_bytes(b"secret")
        assert await hash_dir(d) == h_before

    @pytest.mark.asyncio
    async def test_excludes_hidden_subdirectories(self, tmp_path: Path) -> None:
        d = tmp_path / "dataset"
        d.mkdir()
        (d / "data.csv").write_bytes(b"a,b\n1,2\n")
        h_before = await hash_dir(d)
        hidden = d / ".cache"
        hidden.mkdir()
        (hidden / "tmp.bin").write_bytes(b"cached")
        assert await hash_dir(d) == h_before

    @pytest.mark.asyncio
    async def test_adding_file_changes_hash(self, tmp_path: Path) -> None:
        d = tmp_path / "dataset"
        d.mkdir()
        (d / "a.txt").write_bytes(b"A")
        h_before = await hash_dir(d)
        (d / "b.txt").write_bytes(b"B")
        assert await hash_dir(d) != h_before

    @pytest.mark.asyncio
    async def test_hash_is_order_independent(self, tmp_path: Path) -> None:
        d1 = tmp_path / "d1"
        d2 = tmp_path / "d2"
        d1.mkdir()
        d2.mkdir()
        (d1 / "a.txt").write_bytes(b"AAA")
        (d1 / "b.txt").write_bytes(b"BBB")
        (d2 / "b.txt").write_bytes(b"BBB")
        (d2 / "a.txt").write_bytes(b"AAA")
        assert await hash_dir(d1) == await hash_dir(d2)

    @pytest.mark.asyncio
    async def test_nested_files_included(self, tmp_path: Path) -> None:
        d = tmp_path / "dataset"
        sub = d / "sub"
        sub.mkdir(parents=True)
        (d / "root.txt").write_bytes(b"root")
        (sub / "nested.txt").write_bytes(b"nested")
        h = await hash_dir(d)
        (sub / "nested.txt").write_bytes(b"changed")
        assert await hash_dir(d) != h


class TestHashPath:
    @pytest.mark.asyncio
    async def test_dispatches_to_hash_file_for_file(self, tmp_path: Path) -> None:
        f = tmp_path / "data.bin"
        f.write_bytes(b"content")
        assert await hash_path(f) == hash_file(f)

    @pytest.mark.asyncio
    async def test_dispatches_to_hash_dir_for_directory(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_bytes(b"A")
        assert await hash_path(tmp_path) == await hash_dir(tmp_path)
