"""Unit tests for opendatasci._tui.file_refs — pure parsing and path logic."""

from pathlib import Path
from unittest.mock import patch

import pytest

from opendatasci._tui.file_refs import (
    PasteAttachment,
    _FileRef,
    _build_agent_query,
    _build_user_display,
    _discover_files,
    _find_at_fragment,
    _find_slash_fragment,
    _parse_file_refs,
    _split_existing_file_refs,
)


# ---------------------------------------------------------------------------
# _parse_file_refs
# ---------------------------------------------------------------------------


class TestParseFileRefs:
    def test_no_refs_returns_original_text_and_empty_list(self) -> None:
        clean, refs = _parse_file_refs("hello world")
        assert clean == "hello world"
        assert refs == []

    def test_single_ref_extracted(self) -> None:
        clean, refs = _parse_file_refs("describe @data.csv")
        assert len(refs) == 1
        assert refs[0]._path == "data.csv"

    def test_single_ref_removed_from_clean_text(self) -> None:
        clean, refs = _parse_file_refs("describe @data.csv")
        assert "@data.csv" not in clean
        assert "describe" in clean

    def test_multiple_refs_extracted(self) -> None:
        clean, refs = _parse_file_refs("compare @sales.csv @costs.csv")
        assert len(refs) == 2
        paths = {r._path for r in refs}
        assert "sales.csv" in paths
        assert "costs.csv" in paths

    def test_path_with_directory_separator_extracted(self) -> None:
        clean, refs = _parse_file_refs("see @data/output.csv")
        assert len(refs) == 1
        assert refs[0]._path == "data/output.csv"

    def test_ref_only_text_clean_is_empty(self) -> None:
        clean, refs = _parse_file_refs("@data.csv")
        assert clean == ""
        assert len(refs) == 1

    def test_empty_string_returns_empty(self) -> None:
        clean, refs = _parse_file_refs("")
        assert clean == ""
        assert refs == []

    def test_at_sign_at_end_not_parsed_as_ref(self) -> None:
        # "@" alone at end: _FILE_REF_RE matches @(\S+?) — at least one non-space char
        # So a trailing bare "@" with nothing after should not match.
        clean, refs = _parse_file_refs("hello @")
        # "@" alone doesn't match the regex because \S+? requires at least one char
        # but the lookahead (?=\s|$) immediately terminates. Let's check:
        # Pattern: @(\S+?)(?=\s|$) — non-greedy with lookahead
        # "@" has nothing after it → \S+? must match at least one char → no match.
        assert refs == []

    def test_ref_path_preserves_original_case(self) -> None:
        _, refs = _parse_file_refs("see @MyData.CSV")
        assert refs[0]._path == "MyData.CSV"


# ---------------------------------------------------------------------------
# _build_user_display
# ---------------------------------------------------------------------------


class TestBuildUserDisplay:
    def test_text_only_no_refs_returns_escaped_text(self) -> None:
        result = _build_user_display("tell me about the data", [])
        assert "tell me about the data" in result
        assert "[bold" not in result  # no ref markup

    def test_refs_only_no_text_returns_ref_markup(self) -> None:
        refs = [_FileRef("data.csv")]
        result = _build_user_display("", refs)
        assert "data.csv" in result
        assert "[bold" in result

    def test_display_name_is_filename_not_full_path(self) -> None:
        refs = [_FileRef("path/to/data.csv")]
        result = _build_user_display("", refs)
        assert "data.csv" in result
        assert "path/to/" not in result

    def test_refs_and_text_combined(self) -> None:
        refs = [_FileRef("data.csv")]
        result = _build_user_display("describe it", refs)
        assert "data.csv" in result
        assert "describe it" in result

    def test_text_is_rich_escaped(self) -> None:
        # Special Rich markup characters in text must be escaped so they are
        # rendered literally, not interpreted as markup.
        result = _build_user_display("[bold]not markup[/bold]", [])
        # After escaping, the brackets become escaped sequences
        assert "[bold]not markup[/bold]" not in result or "\\[bold]" in result or "\\[" in result

    def test_multiple_refs_all_appear_in_output(self) -> None:
        refs = [_FileRef("a.csv"), _FileRef("b.csv")]
        result = _build_user_display("", refs)
        assert "a.csv" in result
        assert "b.csv" in result


# ---------------------------------------------------------------------------
# _build_agent_query
# ---------------------------------------------------------------------------


class TestBuildAgentQuery:
    def test_no_refs_returns_clean_text_unchanged(self) -> None:
        result = _build_agent_query("describe the data", [])
        assert result == "describe the data"

    def test_single_ref_adds_file_attachment_tag(self) -> None:
        refs = [_FileRef("data.csv")]
        result = _build_agent_query("describe", refs)
        assert "<file_attachment" in result
        assert 'name="data.csv"' in result

    def test_file_attachment_includes_absolute_path(self) -> None:
        refs = [_FileRef("data.csv")]
        result = _build_agent_query("", refs)
        # path attribute must be an absolute path
        import re
        m = re.search(r'path="([^"]+)"', result)
        assert m is not None
        path_val = m.group(1)
        assert Path(path_val).is_absolute()

    def test_empty_text_with_ref_omits_leading_separator(self) -> None:
        refs = [_FileRef("data.csv")]
        result = _build_agent_query("", refs)
        # Should NOT start with "\n\n" (no leading empty text part)
        assert not result.startswith("\n\n")

    def test_multiple_refs_produce_multiple_tags(self) -> None:
        refs = [_FileRef("a.csv"), _FileRef("b.csv")]
        result = _build_agent_query("compare", refs)
        assert result.count("<file_attachment") == 2

    def test_text_and_ref_separated_by_double_newline(self) -> None:
        refs = [_FileRef("data.csv")]
        result = _build_agent_query("describe", refs)
        assert "\n\n" in result

    def test_ref_display_name_uses_filename_only(self) -> None:
        refs = [_FileRef("some/path/to/data.csv")]
        result = _build_agent_query("", refs)
        assert 'name="data.csv"' in result


# ---------------------------------------------------------------------------
# _find_slash_fragment
# ---------------------------------------------------------------------------


class TestFindSlashFragment:
    def test_slash_prefix_with_no_space_returns_text(self) -> None:
        assert _find_slash_fragment("/clear") == "/clear"

    def test_partial_slash_command_returned(self) -> None:
        assert _find_slash_fragment("/cl") == "/cl"

    def test_slash_alone_returned(self) -> None:
        assert _find_slash_fragment("/") == "/"

    def test_slash_with_space_returns_none(self) -> None:
        assert _find_slash_fragment("/clear world") is None

    def test_non_slash_text_returns_none(self) -> None:
        assert _find_slash_fragment("hello") is None

    def test_empty_string_returns_none(self) -> None:
        assert _find_slash_fragment("") is None

    def test_at_sign_text_returns_none(self) -> None:
        assert _find_slash_fragment("@data.csv") is None


# ---------------------------------------------------------------------------
# _find_at_fragment
# ---------------------------------------------------------------------------


class TestFindAtFragment:
    def test_at_at_start_returns_fragment_and_position(self) -> None:
        result = _find_at_fragment("@data.csv")
        assert result is not None
        fragment, at_pos = result
        assert fragment == "data.csv"
        assert at_pos == 0

    def test_at_in_middle_returns_correct_position(self) -> None:
        result = _find_at_fragment("describe @data.csv")
        assert result is not None
        fragment, at_pos = result
        assert fragment == "data.csv"
        assert at_pos == 9

    def test_at_with_trailing_space_stops_fragment_at_space(self) -> None:
        result = _find_at_fragment("@data.csv more text")
        assert result is not None
        fragment, _ = result
        assert fragment == "data.csv"

    def test_no_at_sign_returns_none(self) -> None:
        assert _find_at_fragment("hello world") is None

    def test_empty_string_returns_none(self) -> None:
        assert _find_at_fragment("") is None

    def test_multiple_at_signs_uses_last_one(self) -> None:
        # rfind returns position of last @
        result = _find_at_fragment("@first @second")
        assert result is not None
        fragment, at_pos = result
        assert fragment == "second"
        assert at_pos == 7

    def test_at_with_path_fragment(self) -> None:
        result = _find_at_fragment("@data/out")
        assert result is not None
        fragment, _ = result
        assert fragment == "data/out"

    def test_bare_at_returns_empty_fragment(self) -> None:
        # "@" at end of string: after="", space_pos=-1, fragment=""
        result = _find_at_fragment("hello @")
        assert result is not None
        fragment, at_pos = result
        assert fragment == ""
        assert at_pos == 6


# ---------------------------------------------------------------------------
# _split_existing_file_refs
# ---------------------------------------------------------------------------


class TestSplitExistingFileRefs:
    def test_existing_file_placed_in_existing_list(self, tmp_path: Path) -> None:
        f = tmp_path / "data.csv"
        f.write_text("col,val\n1,2\n")
        ref = _FileRef(str(f))
        existing, missing = _split_existing_file_refs([ref])
        assert len(existing) == 1
        assert len(missing) == 0

    def test_missing_file_placed_in_missing_list(self, tmp_path: Path) -> None:
        ref = _FileRef(str(tmp_path / "nonexistent.csv"))
        existing, missing = _split_existing_file_refs([ref])
        assert len(existing) == 0
        assert len(missing) == 1

    def test_directory_not_treated_as_existing_file(self, tmp_path: Path) -> None:
        sub = tmp_path / "mydir"
        sub.mkdir()
        ref = _FileRef(str(sub))
        existing, missing = _split_existing_file_refs([ref])
        assert len(existing) == 0
        assert len(missing) == 1

    def test_mixed_refs_split_correctly(self, tmp_path: Path) -> None:
        f = tmp_path / "real.csv"
        f.write_text("a,b\n")
        refs = [_FileRef(str(f)), _FileRef(str(tmp_path / "fake.csv"))]
        existing, missing = _split_existing_file_refs(refs)
        assert len(existing) == 1
        assert len(missing) == 1

    def test_empty_list_returns_two_empty_lists(self) -> None:
        existing, missing = _split_existing_file_refs([])
        assert existing == []
        assert missing == []


# ---------------------------------------------------------------------------
# PasteAttachment
# ---------------------------------------------------------------------------


class TestPasteAttachment:
    def test_display_label_single_line(self) -> None:
        attach = PasteAttachment("one line")
        assert attach.display_label == "Text: 1 line"

    def test_display_label_multi_line(self) -> None:
        attach = PasteAttachment("line1\nline2\nline3")
        assert attach.display_label == "Text: 3 lines"

    def test_display_label_two_lines_uses_plural(self) -> None:
        attach = PasteAttachment("a\nb")
        assert "lines" in attach.display_label

    def test_display_label_exactly_one_line_uses_singular(self) -> None:
        attach = PasteAttachment("no newlines here")
        assert "1 line" in attach.display_label

    def test_xml_tag_wraps_content(self) -> None:
        attach = PasteAttachment("def foo(): pass")
        tag = attach.xml_tag
        assert "<pasted_content>" in tag
        assert "def foo(): pass" in tag
        assert "</pasted_content>" in tag

    def test_xml_tag_content_is_on_inner_line(self) -> None:
        attach = PasteAttachment("my code")
        tag = attach.xml_tag
        # Format is: <pasted_content>\n{content}\n</pasted_content>
        assert tag.startswith("<pasted_content>\n")
        assert tag.endswith("\n</pasted_content>")

    def test_pill_markup_contains_display_label(self) -> None:
        attach = PasteAttachment("a\nb\nc")
        assert attach.display_label in attach.pill_markup

    def test_pill_markup_is_rich_formatted(self) -> None:
        attach = PasteAttachment("code")
        assert "[bold" in attach.pill_markup

    def test_line_count_counts_newlines_plus_one(self) -> None:
        content = "a\nb\nc\nd"  # 3 newlines → 4 lines
        attach = PasteAttachment(content)
        assert attach._line_count == 4


# ---------------------------------------------------------------------------
# _discover_files — caching and prefix matching
# ---------------------------------------------------------------------------


class TestDiscoverFiles:
    def test_returns_matching_files_in_dir(self, tmp_path: Path, monkeypatch) -> None:
        (tmp_path / "sales.csv").write_text("")
        (tmp_path / "costs.csv").write_text("")
        (tmp_path / "report.xlsx").write_text("")
        monkeypatch.chdir(tmp_path)

        matches = _discover_files("sal")
        assert "sales.csv" in matches

    def test_prefix_match_is_case_insensitive(self, tmp_path: Path, monkeypatch) -> None:
        (tmp_path / "Sales.csv").write_text("")
        monkeypatch.chdir(tmp_path)

        matches = _discover_files("sal")
        assert any("Sales.csv" in m for m in matches)

    def test_hidden_files_excluded_without_dot_prefix(self, tmp_path: Path, monkeypatch) -> None:
        (tmp_path / ".hidden").write_text("")
        (tmp_path / "visible.csv").write_text("")
        monkeypatch.chdir(tmp_path)

        matches = _discover_files("")
        names = [Path(m).name for m in matches]
        assert ".hidden" not in names

    def test_hidden_files_included_with_dot_prefix(self, tmp_path: Path, monkeypatch) -> None:
        (tmp_path / ".env").write_text("")
        monkeypatch.chdir(tmp_path)

        matches = _discover_files(".")
        assert any(".env" in m for m in matches)

    def test_results_capped_at_ten(self, tmp_path: Path, monkeypatch) -> None:
        for i in range(15):
            (tmp_path / f"file{i}.csv").write_text("")
        monkeypatch.chdir(tmp_path)

        matches = _discover_files("")
        assert len(matches) <= 10

    def test_directories_suffixed_with_slash(self, tmp_path: Path, monkeypatch) -> None:
        sub = tmp_path / "subdir"
        sub.mkdir()
        monkeypatch.chdir(tmp_path)

        matches = _discover_files("sub")
        assert any(m.endswith("/") for m in matches)

    def test_nonexistent_dir_returns_empty(self, tmp_path: Path) -> None:
        # Fragment points to a non-existent subdirectory
        matches = _discover_files(str(tmp_path / "no_such_dir") + "/file")
        assert matches == []
