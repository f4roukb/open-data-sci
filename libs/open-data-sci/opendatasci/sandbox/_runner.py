"""Injected into the SRT sandbox session directory and executed as a subprocess.

Reads user code from an env-var, executes it inside a fresh namespace (no
state carries over from any prior call), and emits a single JSON payload to
stdout.
"""

import base64
import io
import json
import os
import sys
import traceback
from pathlib import Path
from typing import TextIO

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore[assignment]

WORKSPACE = os.environ.get("OPENDATASCI_WORKSPACE", "/tmp/opendatasci_workspace")
code = base64.b64decode(os.environ["OPENDATASCI_CODE_B64"]).decode("utf-8")

# Duplicated verbatim from ``opendatasci.sandbox.base.PAYLOAD_SENTINEL`` --
# this script must stay import-free of the ``opendatasci`` package (it runs
# standalone inside the sandboxed subprocess, see module docstring), so the
# shared framing marker is a literal here rather than an import.
PAYLOAD_SENTINEL = "###OPENDATASCI_PAYLOAD###"


class _TeeStdout(io.TextIOBase):
    """Writes through to the real stdout immediately (so a parent process
    reading the pipe sees progress as it happens) while also buffering
    everything, so the final payload's ``stdout`` field still carries the
    full transcript for callers that don't stream.

    Subclasses ``TextIOBase`` (rather than composing ``io.StringIO``
    directly) so that user code calling other stream methods (``isatty()``,
    ``writable()``, etc. -- e.g. from ``tqdm`` or ``rich``) gets the usual
    harmless defaults instead of ``AttributeError``.
    """

    def __init__(self, real_stdout: TextIO) -> None:
        self._real = real_stdout
        self._buffer = io.StringIO()

    def write(self, text: str) -> int:
        self._buffer.write(text)
        self._real.write(text)
        self._real.flush()
        return len(text)

    def flush(self) -> None:
        self._real.flush()

    def writable(self) -> bool:
        return True

    def getvalue(self) -> str:
        return self._buffer.getvalue()


namespace: dict[str, object] = {}

workspacedir = Path(WORKSPACE)
opendatasci_directory = workspacedir / ".opendatasci"
opendatasci_directory.mkdir(parents=True, exist_ok=True)
os.chdir(str(workspacedir))

saved_results: dict[str, object] = {}


def save_result(name: str, value: object) -> None:
    saved_results[name] = value


namespace.update(
    {
        "workspacedir": workspacedir,
        "opendatasci_directory": opendatasci_directory,
        "save_result": save_result,
    }
)

skip_keys = {"workspacedir", "opendatasci_directory", "save_result", "__builtins__"}

original_stdout = sys.stdout
tee = _TeeStdout(original_stdout)
sys.stdout = tee
sys.stdin = io.StringIO("")
try:
    exec(compile(code, "<opendatasci>", "exec"), namespace)  # noqa: S102
    output_value = namespace.pop("result", None)

    var_info = {}
    for key, value in namespace.items():
        if key.startswith("_") or key in skip_keys:
            continue

        if pd is not None and isinstance(value, pd.DataFrame):
            description = f"DataFrame {value.shape}"
        elif isinstance(value, (list, dict)):
            description = f"{type(value).__name__} (len={len(value)})"
        else:
            description = type(value).__name__

        var_info[key] = description

    payload = {
        "success": True,
        "stdout": tee.getvalue(),
        "result": repr(output_value) if output_value is not None else None,
        "var_info": var_info,
        "saved_results": {k: repr(v) for k, v in saved_results.items()},
    }
except Exception as exc:
    payload = {
        "success": False,
        "stdout": tee.getvalue(),
        "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        "var_info": {},
        "saved_results": {},
    }
finally:
    sys.stdout = original_stdout

print(f"{PAYLOAD_SENTINEL}{json.dumps(payload)}")
