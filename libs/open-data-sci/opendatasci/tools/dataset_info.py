"""Dataset context tools and profiling code generation.

Exposes:
- ``build_profile_code(path)``    – sandbox-ready profiling snippet generator.
- ``get_load_dataset_info_tool(session)``  – load persistent dataset info.
- ``get_profile_dataset_tool(session)``    – profile a dataset and cache the card.
- ``get_data_context_tools(session)``      – all three tools as a list.
"""

from langchain_core.tools import BaseTool, tool

from opendatasci.context.base import BaseContextStore
from opendatasci.sandbox.base import BaseSandbox

# ---------------------------------------------------------------------------
# Profiling code generation
# ---------------------------------------------------------------------------
# All curly-braces that belong to *Python* code (f-strings, dict literals,
# format specs) are doubled so they survive the .format(path_repr=...) call.
# Only ``{path_repr}`` is a real format placeholder.
# ---------------------------------------------------------------------------

_PROFILE_CODE_TEMPLATE = """\
import datetime
import pathlib

import pandas as pd

_path = pathlib.Path({path_repr})
_ext = _path.suffix.lower()

_LOADERS = {{
    ".csv":    lambda p: pd.read_csv(p, low_memory=False),
    ".tsv":    lambda p: pd.read_csv(p, sep="\\t", low_memory=False),
    ".txt":    lambda p: pd.read_csv(p, low_memory=False),
    ".parquet": lambda p: pd.read_parquet(p),
    ".pq":     lambda p: pd.read_parquet(p),
    ".json":   lambda p: pd.read_json(p),
    ".jsonl":  lambda p: pd.read_json(p, lines=True),
    ".ndjson": lambda p: pd.read_json(p, lines=True),
    ".xlsx":   lambda p: pd.read_excel(p),
    ".xls":    lambda p: pd.read_excel(p),
    ".xlsm":   lambda p: pd.read_excel(p),
    ".feather": lambda p: pd.read_feather(p),
    ".ftr":    lambda p: pd.read_feather(p),
    ".pkl":    lambda p: pd.read_pickle(p),
}}

_loader = _LOADERS.get(_ext)
try:
    if _loader is not None:
        _df = _loader(_path)
    else:
        _df = pd.read_csv(_path, low_memory=False)
except Exception as _load_err:
    result = f"__PROFILE_SKIP__Cannot load {{_path.name}} ({{type(_load_err).__name__}}: {{_load_err}})"
else:
    _rows, _cols = _df.shape
    _mem_mb = _df.memory_usage(deep=True).sum() / 1_048_576
    _dupes = int(_df.duplicated().sum())
    _ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # --- Per-column unique counts (computed once, reused below) ---
    _unique_counts = {{col: int(_df[col].nunique(dropna=True)) for col in _df.columns}}

    # --- Pandera schema inference ---
    # Used for (a) nullable-integer detection and (b) the Inferred Schema section.
    _pa_schema = None
    _pa_nullable = {{}}   # col -> bool
    try:
        import pandera as pa
        _pa_schema = pa.infer_schema(_df)
        _pa_nullable = {{n: c.nullable for n, c in _pa_schema.columns.items()}}
    except Exception:
        pass

    # --- Data quality flags ---
    _dq_flags = []
    for _col in _df.columns:
        _nuniq   = _unique_counts[_col]
        _nonnull = int(_df[_col].notna().sum())
        _null_pct = ((_rows - _nonnull) / _rows * 100) if _rows > 0 else 0.0
        _dtype_s  = str(_df[_col].dtype)

        if _nuniq == 1:
            _dq_flags.append((_col, "Constant", "Only 1 unique value across all rows"))
        if _rows > 0 and _nuniq == _rows and _null_pct == 0:
            _dq_flags.append((_col, "Likely ID", "100% unique, no nulls"))
        elif _rows > 0 and _nuniq / _rows > 0.95 and _null_pct < 1 and _dtype_s == "object":
            _dq_flags.append((_col, "High cardinality", f"{{_nuniq / _rows * 100:.1f}}% unique values"))
        if _null_pct > 50:
            _dq_flags.append((_col, "High nulls", f"{{_null_pct:.1f}}% missing"))
        # Pandera marks these nullable=True; cross-check that all non-null values are
        # whole numbers to confirm the column is really an integer column with nulls.
        if _dtype_s.startswith("float") and _pa_nullable.get(_col, False):
            _sample = _df[_col].dropna()
            if len(_sample) > 0 and (_sample % 1 == 0).all():
                _dq_flags.append((
                    _col, "Nullable int",
                    f"Stored as {{_dtype_s}}; all non-null values are integers — "
                    "consider casting to a nullable integer dtype",
                ))

    _lines = [
        f"# Dataset Profile: {{_path.name}}",
        "",
        f"**Profiled:** {{_ts}}  ",
        f"**Path:** `{{_path}}`",
        "",
        "## Overview",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Rows | {{_rows:,}} |",
        f"| Columns | {{_cols:,}} |",
        f"| Memory | {{_mem_mb:.2f}} MB |",
        f"| Duplicate rows | {{_dupes:,}} |",
        "",
        "## Columns",
        "",
        "| Column | Type | Non-null | Null % | Unique | Unique % |",
        "|--------|------|----------|--------|--------|----------|",
    ]

    for _col in _df.columns:
        _dtype   = str(_df[_col].dtype)
        _nonnull = int(_df[_col].notna().sum())
        _null_pct = ((_rows - _nonnull) / _rows * 100) if _rows > 0 else 0.0
        _nuniq   = _unique_counts[_col]
        _uniq_pct = (_nuniq / _rows * 100) if _rows > 0 else 0.0
        _lines.append(
            f"| `{{_col}}` | {{_dtype}} | {{_nonnull:,}} | {{_null_pct:.1f}}% "
            f"| {{_nuniq:,}} | {{_uniq_pct:.1f}}% |"
        )

    # --- Data quality flags ---
    if _dq_flags:
        _lines += [
            "",
            "## Data Quality Flags",
            "",
            "| Column | Flag | Detail |",
            "|--------|------|--------|",
        ]
        for _dq_col, _dq_flag, _dq_detail in _dq_flags:
            _lines.append(f"| `{{_dq_col}}` | {{_dq_flag}} | {{_dq_detail}} |")

    # --- Numeric summary ---
    _num_cols = _df.select_dtypes(include="number").columns.tolist()
    if _num_cols:
        _lines += [
            "",
            "## Numeric Summary",
            "",
            "| Column | Mean | Std | Min | p25 | p50 | p75 | Max |",
            "|--------|------|-----|-----|-----|-----|-----|-----|",
        ]
        _desc = _df[_num_cols].describe(percentiles=[0.25, 0.5, 0.75])

        def _fmt(v):
            if pd.isna(v):
                return "—"
            return f"{{v:,.2f}}" if abs(v) < 1_000_000 else f"{{v:.3e}}"

        for _col in _num_cols:
            _lines.append(
                f"| `{{_col}}` "
                f"| {{_fmt(_desc.loc['mean', _col])}} "
                f"| {{_fmt(_desc.loc['std', _col])}} "
                f"| {{_fmt(_desc.loc['min', _col])}} "
                f"| {{_fmt(_desc.loc['25%', _col])}} "
                f"| {{_fmt(_desc.loc['50%', _col])}} "
                f"| {{_fmt(_desc.loc['75%', _col])}} "
                f"| {{_fmt(_desc.loc['max', _col])}} |"
            )

    # --- Top categoricals ---
    try:
        _cat_dtypes = ["object", "category", "string"]
        _cat_cols = _df.select_dtypes(include=_cat_dtypes).columns.tolist()
    except Exception:
        _cat_cols = _df.select_dtypes(include=["object", "category"]).columns.tolist()

    if _cat_cols:
        _lines.append("")
        _lines.append("## Top Categoricals")
        for _col in _cat_cols:
            _vc = _df[_col].value_counts().head(5)
            if _vc.empty:
                continue
            _lines += [
                "",
                f"### `{{_col}}`",
                "",
                "| Value | Count | % |",
                "|-------|-------|---|",
            ]
            for _val, _cnt in _vc.items():
                _pct = (_cnt / _rows * 100) if _rows > 0 else 0.0
                _display = str(_val)
                if len(_display) > 40:
                    _display = _display[:37] + "..."
                _lines.append(f"| {{_display!r}} | {{_cnt:,}} | {{_pct:.1f}}% |")

    # --- Pandera inferred schema (dtype + nullable per column) ---
    if _pa_schema is not None:
        _lines += [
            "",
            "## Inferred Schema (pandera)",
            "",
            "| Column | Inferred dtype | Nullable |",
            "|--------|----------------|----------|",
        ]
        for _pa_col_name, _pa_col in _pa_schema.columns.items():
            _pa_col_nullable = "yes" if _pa_col.nullable else "no"
            _lines.append(
                f"| `{{_pa_col_name}}` | {{_pa_col.dtype}} | {{_pa_col_nullable}} |"
            )

    result = "\\n".join(_lines) + "\\n"
"""


def build_profile_code(path: str) -> str:
    """Return a sandbox-ready Python snippet that profiles the dataset at *path*.

    The snippet sets ``result`` to the Markdown profile card on success, or to
    a ``__PROFILE_SKIP__<reason>`` sentinel string when the file cannot be
    loaded (so the caller can surface a helpful message without saving garbage).

    Args:
        path: Absolute filesystem path to the dataset file.  Must be absolute
              so the snippet runs correctly regardless of the sandbox CWD.
    """
    return _PROFILE_CODE_TEMPLATE.format(path_repr=repr(path))


# ---------------------------------------------------------------------------
# LangChain tools
# ---------------------------------------------------------------------------


def create_read_dataset_info_tools(context: BaseContextStore | None) -> list[BaseTool]:
    """Return the ``read_dataset_info`` tool bound to *context*."""

    @tool
    async def read_dataset_info(path: str, summary: str, communication: str) -> str:
        """Load all accumulated knowledge about a dataset — profile card and agent notes.

        Returns a Markdown string with two sections:
        - ``# DATASET PROFILING`` — auto-generated profile card (shape, dtypes, null rates,
          numeric summary, top categoricals).
        - ``# DATASET NOTES`` — cumulative agent notes from all prior sessions. Primary source
          of institutional knowledge; read carefully before exploring.

        # When to use this tool
        - Always, before writing any code that reads or processes a dataset.
        - Even if you profiled the dataset earlier in the session — notes may have been
          updated by ``update_dataset_info`` since then.

        Args:
            path:          Absolute or relative path to the dataset file or directory.
            summary:       3-4 word status label (e.g. "Reading train.csv info").
            communication: Brief message to the user about what you're doing
                           (e.g. "Let me read existing notes about the dataset.").
        """
        if context is None:
            return "Error: No workspace path available."
        try:
            return await context.read_dataset_info(path)
        except FileNotFoundError as exc:
            return f"Error: {exc}"
        except Exception as exc:
            return f"Error loading dataset info: {type(exc).__name__}: {exc}"

    return [read_dataset_info]


def create_profile_dataset_tools(
    context: BaseContextStore | None, sandbox: BaseSandbox, persist: bool = True
) -> list[BaseTool]:
    """Return the ``profile_dataset`` tool bound to *context* and *sandbox*."""

    @tool
    async def profile_dataset(path: str, summary: str, communication: str) -> str:
        """Auto-profile a dataset and return its card (shape, dtypes, null rates, distributions).

        Profiles are cached — subsequent calls return the existing card without re-scanning.
        The card covers: shape, dtypes, null rates, numeric summary, top categoricals,
        data-quality flags, and inferred schema.

        # When to use this tool
        - Once per new dataset, before exploring it.
        - When ``read_dataset_info`` returns an empty or stale profile for a dataset.

        # When NOT to use this tool
        - When a profile was already returned by ``read_dataset_info`` this session — it is current.

        Args:
            path:          Absolute or relative path to the dataset file or directory.
            summary:       3-4 word status label (e.g. "Profiling train.csv").
            communication: Brief message to the user about what you're doing
                           (e.g. "Let me profile the dataset to understand its structure.").
        """
        if context is None:
            return "Error: No workspace path available."
        try:
            wc = context
            resolved, hash_hex, existing = await wc.get_profile_info(path)

            if existing is not None:
                return existing

            code = build_profile_code(resolved)
            exec_result = await sandbox.execute(code)

            if not exec_result.success:
                return f"Profiling failed: {exec_result.error}"

            if exec_result.output is None:
                return "Profiling produced no output — the dataset may be empty."

            content = str(exec_result.output)
            if content.startswith("__PROFILE_SKIP__"):
                return content[len("__PROFILE_SKIP__") :]

            if persist:
                wc.save_dataset_profile(hash_hex, content)
            return content

        except FileNotFoundError as exc:
            return f"Error: {exc}"
        except Exception as exc:
            return f"Error profiling dataset: {type(exc).__name__}: {exc}"

    return [profile_dataset]


def create_update_dataset_info_tools(context: BaseContextStore | None) -> list[BaseTool]:
    """Return the ``update_dataset_info`` tool bound to *context*."""

    @tool
    async def update_dataset_info(path: str, update: str, merge: bool = False) -> str:
        """Persist findings, observations, and decisions for a dataset across sessions.

        This is how knowledge carries forward — everything written here is surfaced
        automatically next session via ``read_dataset_info``. Skipping means the
        next session starts blind to the new knowledge and context about you've gathered
        about the data and tasks you've tackled around it.

        # When to use this tool
        - After every turn that touches a dataset: findings, quality issues, surprises,
            failed approaches, user decisions, and hypotheses to revisit.
        - Err on the side of capturing more — even minor or confirmatory findings.

        Args:
            path:   Absolute or relative path to the dataset file or directory.
            update: Markdown content to write or append. Make it structured.
            merge:  ``True`` to append to existing notes (default);
                    ``False`` to overwrite all prior notes.
        """
        if context is None:
            return "Error: No workspace path available."
        try:
            return await context.update_dataset_info(path, update, merge=merge)
        except FileNotFoundError as exc:
            return f"Error: {exc}"
        except Exception as exc:
            return f"Error updating dataset info: {type(exc).__name__}: {exc}"

    return [update_dataset_info]


def create_data_context_tools(
    context: BaseContextStore | None, sandbox: BaseSandbox, persist: bool = True
) -> list[BaseTool]:
    """Return dataset-context tools bound to *context* and *sandbox*.

    Args:
        context: I/O boundary for dataset notes and profiles.
        sandbox:      Execution sandbox used by ``profile_dataset``.
        persist:      When ``False``, ``update_dataset_info`` is excluded and
                      ``profile_dataset`` will not write profiles to disk.
    """
    tools = [
        *create_read_dataset_info_tools(context),
        *create_profile_dataset_tools(context, sandbox, persist=persist),
    ]
    if persist:
        tools.extend(create_update_dataset_info_tools(context))
    return tools
