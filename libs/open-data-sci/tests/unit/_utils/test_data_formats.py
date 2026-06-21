"""Unit tests for opendatasci._utils.data_formats — detect_format()."""


from pathlib import Path

import pytest

from opendatasci._utils.data_formats import (
    ALL_SUPPORTED_EXTENSIONS,
    ARCHIVE_EXTENSIONS,
    EXCEL_EXTENSIONS,
    LOADABLE_EXTENSIONS,
    detect_format,
)


class TestExtensionSetMembership:
    def test_excel_extensions_excluded_from_archives(self) -> None:
        assert not (EXCEL_EXTENSIONS & ARCHIVE_EXTENSIONS)

    def test_archive_extensions_excluded_from_loadable(self) -> None:
        assert not (LOADABLE_EXTENSIONS & ARCHIVE_EXTENSIONS)

    def test_all_supported_is_union(self) -> None:
        assert ALL_SUPPORTED_EXTENSIONS == LOADABLE_EXTENSIONS | ARCHIVE_EXTENSIONS


class TestDetectFormatExcel:
    @pytest.mark.parametrize("ext", [".xlsx", ".xls", ".xlsm", ".xlsb"])
    def test_excel_variants_detected(self, ext: str) -> None:
        fmt, compression = detect_format(Path(f"book{ext}"))
        assert fmt == "excel"
        assert compression is None

    def test_uppercase_extension_normalised(self) -> None:
        fmt, _ = detect_format(Path("BOOK.XLSX"))
        assert fmt == "excel"


class TestDetectFormatCSV:
    @pytest.mark.parametrize("ext", [".csv", ".tsv"])
    def test_csv_variants_detected(self, ext: str) -> None:
        fmt, compression = detect_format(Path(f"data{ext}"))
        assert fmt == "csv"
        assert compression is None


class TestDetectFormatJSON:
    def test_json_extension(self) -> None:
        assert detect_format(Path("a.json")) == ("json", None)

    @pytest.mark.parametrize("ext", [".jsonl", ".ndjson"])
    def test_line_delimited_json_extension(self, ext: str) -> None:
        # jsonl and ndjson must be distinguished from regular json.
        assert detect_format(Path(f"a{ext}")) == ("jsonl", None)


class TestDetectFormatBinaryFormats:
    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("a.parquet", "parquet"),
            ("a.pq", "parquet"),
            ("a.orc", "orc"),
            ("a.feather", "feather"),
            ("a.arrow", "feather"),
            ("a.h5", "hdf"),
            ("a.hdf", "hdf"),
            ("a.hdf5", "hdf"),
            ("a.pkl", "pickle"),
            ("a.pickle", "pickle"),
            ("a.xml", "xml"),
        ],
    )
    def test_format_dispatch(self, filename: str, expected: str) -> None:
        assert detect_format(Path(filename))[0] == expected


class TestDetectFormatStatisticalFormats:
    def test_stata(self) -> None:
        assert detect_format(Path("a.dta"))[0] == "stata"

    @pytest.mark.parametrize("ext", [".sas7bdat", ".xpt"])
    def test_sas(self, ext: str) -> None:
        assert detect_format(Path(f"a{ext}"))[0] == "sas"

    @pytest.mark.parametrize("ext", [".sav", ".zsav"])
    def test_spss(self, ext: str) -> None:
        assert detect_format(Path(f"a{ext}"))[0] == "spss"


class TestDetectFormatTextFormats:
    @pytest.mark.parametrize(
        "ext",
        [".txt", ".md", ".log", ".yml", ".yaml", ".toml", ".ini", ".cfg", ".conf", ".rst", ".text"],
    )
    def test_text_extensions(self, ext: str) -> None:
        assert detect_format(Path(f"notes{ext}"))[0] == "text"


class TestDetectFormatArchive:
    def test_zip_is_archive(self) -> None:
        # .zip is treated as an archive container, not as data.
        assert detect_format(Path("bundle.zip")) == ("zip", None)


class TestDetectFormatCompression:
    @pytest.mark.parametrize(
        "filename,expected_fmt,expected_compression",
        [
            ("data.csv.gz", "csv", "gz"),
            ("data.csv.bz2", "csv", "bz2"),
            ("data.csv.xz", "csv", "xz"),
            ("data.csv.zst", "csv", "zst"),
            ("data.json.gz", "json", "gz"),
            ("logs.jsonl.gz", "jsonl", "gz"),
            ("data.parquet.gz", "parquet", "gz"),
        ],
    )
    def test_compression_strips_outer_extension(
        self, filename: str, expected_fmt: str, expected_compression: str
    ) -> None:
        fmt, compression = detect_format(Path(filename))
        assert fmt == expected_fmt
        assert compression == expected_compression

    def test_compression_only_no_known_inner_returns_unknown_with_compression(self) -> None:
        # ``mystery.gz`` strips .gz, leaves no recognisable extension.
        fmt, compression = detect_format(Path("mystery.gz"))
        assert fmt == "unknown"
        assert compression == "gz"


class TestDetectFormatUnknown:
    def test_unknown_extension_returns_unknown(self) -> None:
        assert detect_format(Path("data.weirdformat")) == ("unknown", None)

    def test_no_extension_returns_unknown(self) -> None:
        assert detect_format(Path("README")) == ("unknown", None)
