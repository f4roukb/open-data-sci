"""
File-format detection and extension registry.

Centralising format metadata here means ``LocalWorkspace`` does not grow when new
formats are added — only this module changes.  Any other module that needs to
know which extensions are supported can import from here rather than
re-declaring its own sets.
"""

from pathlib import Path
from typing import Literal

EXCEL_EXTENSIONS: frozenset[str] = frozenset({".xlsx", ".xls", ".xlsm", ".xlsb"})
CSV_EXTENSIONS: frozenset[str] = frozenset({".csv", ".tsv"})
JSON_EXTENSIONS: frozenset[str] = frozenset({".json", ".jsonl", ".ndjson"})
PARQUET_EXTENSIONS: frozenset[str] = frozenset({".parquet", ".pq"})
ORC_EXTENSIONS: frozenset[str] = frozenset({".orc"})
FEATHER_EXTENSIONS: frozenset[str] = frozenset({".feather", ".arrow"})
HDF_EXTENSIONS: frozenset[str] = frozenset({".h5", ".hdf", ".hdf5"})
PICKLE_EXTENSIONS: frozenset[str] = frozenset({".pkl", ".pickle"})
XML_EXTENSIONS: frozenset[str] = frozenset({".xml"})
STATA_EXTENSIONS: frozenset[str] = frozenset({".dta"})
SAS_EXTENSIONS: frozenset[str] = frozenset({".sas7bdat", ".xpt"})
SPSS_EXTENSIONS: frozenset[str] = frozenset({".sav", ".zsav"})
TEXT_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".txt",
        ".md",
        ".log",
        ".yml",
        ".yaml",
        ".toml",
        ".ini",
        ".cfg",
        ".conf",
        ".rst",
        ".text",
    }
)
ARCHIVE_EXTENSIONS: frozenset[str] = frozenset({".zip", ".gz", ".bz2", ".xz", ".zst"})

# All extensions that can be loaded directly as data (archives are containers,
# not data files themselves, so they are tracked separately).
LOADABLE_EXTENSIONS: frozenset[str] = (
    EXCEL_EXTENSIONS
    | CSV_EXTENSIONS
    | JSON_EXTENSIONS
    | PARQUET_EXTENSIONS
    | ORC_EXTENSIONS
    | FEATHER_EXTENSIONS
    | HDF_EXTENSIONS
    | PICKLE_EXTENSIONS
    | XML_EXTENSIONS
    | STATA_EXTENSIONS
    | SAS_EXTENSIONS
    | SPSS_EXTENSIONS
    | TEXT_EXTENSIONS
)

ALL_SUPPORTED_EXTENSIONS: frozenset[str] = LOADABLE_EXTENSIONS | ARCHIVE_EXTENSIONS

FileFormat = Literal[
    "excel",
    "csv",
    "json",
    "jsonl",
    "parquet",
    "orc",
    "feather",
    "hdf",
    "pickle",
    "xml",
    "stata",
    "sas",
    "spss",
    "text",
    "zip",
    "unknown",
]


def detect_format(path: Path) -> tuple[FileFormat, str | None]:
    """Detect the file format and compression of *path*.

    Returns:
        A ``(format, compression)`` tuple.  ``compression`` is the compression
        scheme without a leading dot (e.g. ``"gz"``) or ``None`` for
        uncompressed files.

    Examples::

        detect_format(Path("data.csv"))        # ("csv", None)
        detect_format(Path("data.csv.gz"))     # ("csv", "gz")
        detect_format(Path("archive.zip"))     # ("zip", None)
    """
    suffix = path.suffix.lower()
    compression: str | None = None
    actual_suffix = suffix

    if suffix in {".gz", ".bz2", ".xz", ".zst"}:
        compression = suffix[1:]  # strip leading dot
        actual_suffix = Path(path.stem).suffix.lower()

    if suffix == ".zip":
        return "zip", None
    if actual_suffix in EXCEL_EXTENSIONS:
        return "excel", compression
    if actual_suffix in CSV_EXTENSIONS:
        return "csv", compression
    if actual_suffix in {".jsonl", ".ndjson"}:
        return "jsonl", compression
    if actual_suffix in JSON_EXTENSIONS:
        return "json", compression
    if actual_suffix in PARQUET_EXTENSIONS:
        return "parquet", compression
    if actual_suffix in ORC_EXTENSIONS:
        return "orc", compression
    if actual_suffix in FEATHER_EXTENSIONS:
        return "feather", compression
    if actual_suffix in HDF_EXTENSIONS:
        return "hdf", compression
    if actual_suffix in PICKLE_EXTENSIONS:
        return "pickle", compression
    if actual_suffix in XML_EXTENSIONS:
        return "xml", compression
    if actual_suffix in STATA_EXTENSIONS:
        return "stata", compression
    if actual_suffix in SAS_EXTENSIONS:
        return "sas", compression
    if actual_suffix in SPSS_EXTENSIONS:
        return "spss", compression
    if actual_suffix in TEXT_EXTENSIONS:
        return "text", compression

    return "unknown", compression
