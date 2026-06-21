"""Injected into the SRT sandbox session directory and executed as a subprocess.

Reads user code from an env-var, executes it inside a persistent namespace,
and emits a single JSON payload to stdout.
"""

import base64
import io
import json
import os
import pickle
import sys
import traceback
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore[assignment]

STATE_PATH = os.environ.get("OPENDATASCI_STATE_PATH", "/tmp/opendatasci_state.pkl")
WORKSPACE = os.environ.get("OPENDATASCI_WORKSPACE", "/tmp/opendatasci_workspace")
RESULTS_KEY = "__opendatasci_results__"
code = base64.b64decode(os.environ["OPENDATASCI_CODE_B64"]).decode("utf-8")

# Trust note: state.pkl is a per-session, sandbox-private temp file written only
# by this runner, so unpickling it is safe today. If state ever becomes shared
# or persisted across sessions (e.g. a microservice port), this load() becomes
# an arbitrary-code-execution vector and must be replaced with a safe format.
try:
    with open(STATE_PATH, "rb") as fh:
        namespace = pickle.load(fh)
except FileNotFoundError:
    namespace = {}

workspacedir = Path(WORKSPACE)
opendatasci_directory = workspacedir / ".opendatasci"
opendatasci_directory.mkdir(parents=True, exist_ok=True)
os.chdir(str(workspacedir))

saved_results = namespace.pop(RESULTS_KEY, {})


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
captured = io.StringIO()

original_stdout = sys.stdout
sys.stdout = captured
sys.stdin = io.StringIO("")
try:
    exec(compile(code, "<opendatasci>", "exec"), namespace)  # noqa: S102
    output_value = namespace.pop("result", None)

    clean_ns = {}
    var_info = {}
    dropped = []

    for key, value in namespace.items():
        if key.startswith("_") or key in skip_keys:
            continue

        if pd is not None and isinstance(value, pd.DataFrame):
            description = f"DataFrame {value.shape}"
        elif isinstance(value, (list, dict)):
            description = f"{type(value).__name__} (len={len(value)})"
        else:
            description = type(value).__name__

        try:
            pickle.dumps(value)
        except Exception:
            dropped.append(key)
            continue

        clean_ns[key] = value
        var_info[key] = description

    clean_ns[RESULTS_KEY] = saved_results
    with open(STATE_PATH, "wb") as fh_out:
        pickle.dump(clean_ns, fh_out)

    payload = {
        "success": True,
        "stdout": captured.getvalue(),
        "result": repr(output_value) if output_value is not None else None,
        "var_info": var_info,
        "saved_results": {k: repr(v) for k, v in saved_results.items()},
        "dropped_vars": dropped,
    }
except Exception as exc:
    payload = {
        "success": False,
        "stdout": captured.getvalue(),
        "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        "var_info": {},
        "saved_results": {},
        "dropped_vars": [],
    }
finally:
    sys.stdout = original_stdout

print(json.dumps(payload))
