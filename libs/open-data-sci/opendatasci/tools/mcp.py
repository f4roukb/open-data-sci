"""MCP (Model Context Protocol) adapter: fetch tools from MCP servers and wrap as LangChain Tools."""

import json
import sys
from pathlib import Path
from typing import Any, Optional

import httpx
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import Field, create_model

OPENDATASCI_DIRNAME = ".opendatasci"
_MCP_CONFIG_FILE = "mcp.json"

_MCP_TIMEOUT = 30.0
_JSONRPC_VERSION = "2.0"

_JSON_SCHEMA_TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _jsonrpc(method: str, params: dict[str, Any] | None = None, req_id: int = 1) -> dict[str, Any]:
    payload: dict[str, Any] = {"jsonrpc": _JSONRPC_VERSION, "id": req_id, "method": method}
    if params is not None:
        payload["params"] = params
    return payload


def _initialize(url: str) -> None:
    """Perform the MCP initialization handshake with the server."""
    payload = _jsonrpc(
        "initialize",
        {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "opendatasci", "version": "1.0"},
        },
    )
    with httpx.Client() as client:
        client.post(url, json=payload, timeout=_MCP_TIMEOUT).raise_for_status()
        # Best-effort initialized notification (servers may not require it)
        notif = {"jsonrpc": _JSONRPC_VERSION, "method": "notifications/initialized"}
        try:
            client.post(url, json=notif, timeout=_MCP_TIMEOUT)
        except Exception:
            pass


def _list_tools(url: str) -> list[dict[str, Any]]:
    """Fetch the tool manifest from an MCP server."""
    payload = _jsonrpc("tools/list", req_id=2)
    with httpx.Client() as client:
        resp = client.post(url, json=payload, timeout=_MCP_TIMEOUT)
        resp.raise_for_status()
    data: dict[str, Any] = resp.json()
    return data.get("result", {}).get("tools", [])  # type: ignore[no-any-return]


def _build_args_model(tool_name: str, input_schema: dict[str, Any]) -> type:
    """Convert a JSON Schema object into a Pydantic model for StructuredTool."""
    properties: dict[str, Any] = input_schema.get("properties", {})
    required: set[str] = set(input_schema.get("required", []))
    fields: dict[str, Any] = {}

    for prop_name, prop in properties.items():
        py_type = _JSON_SCHEMA_TYPE_MAP.get(prop.get("type", "string"), str)
        description = prop.get("description", "")
        if prop_name in required:
            fields[prop_name] = (py_type, Field(description=description))
        else:
            fields[prop_name] = (Optional[py_type], Field(default=None, description=description))

    return create_model(f"_{tool_name}_args", **fields)  # type: ignore[no-any-return]


def _make_mcp_tool(server_url: str, tool_def: dict[str, Any]) -> BaseTool:
    """Wrap a single MCP tool definition as an async LangChain StructuredTool."""
    name: str = tool_def["name"]
    description: str = tool_def.get("description", "")
    input_schema: dict[str, Any] = tool_def.get("inputSchema", {"type": "object", "properties": {}})
    args_model = _build_args_model(name, input_schema)

    async def _call(**kwargs: Any) -> str:
        payload = _jsonrpc("tools/call", {"name": name, "arguments": kwargs}, req_id=3)
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(server_url, json=payload, timeout=_MCP_TIMEOUT)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            return f"Error calling MCP tool '{name}': {type(exc).__name__}: {exc}"

        if data.get("error"):
            err = data["error"]
            return f"MCP error {err.get('code', '')}: {err.get('message', str(err))}"

        content = data.get("result", {}).get("content", [])
        parts = [item.get("text", "") for item in content if item.get("type") == "text"]
        if not parts:
            return json.dumps(data.get("result", {}))
        return "\n".join(parts)

    return StructuredTool.from_function(
        coroutine=_call,
        name=name,
        description=description,
        args_schema=args_model,
    )


def _parse_mcp_servers(data: dict[str, Any]) -> list[tuple[str, str]]:
    """Extract ``(name, url)`` pairs from a decoded ``mcp.json`` document."""
    servers: dict[str, dict[str, str]] = data.get("mcpServers", {})
    return [(name, entry["url"]) for name, entry in servers.items() if "url" in entry]


def _workspace_mcp_config_path(workspace_path: Path) -> Path:
    return workspace_path / OPENDATASCI_DIRNAME / _MCP_CONFIG_FILE


def load_workspace_mcp_servers(workspace_path: Path) -> list[tuple[str, str]]:
    """Read ``(name, url)`` pairs from ``<workspace>/.opendatasci/mcp.json``.

    The file format mirrors the Cursor ``mcp.json`` convention::

        {
            "mcpServers": {
                "my-server": { "url": "http://localhost:8080" },
                "another":   { "url": "http://localhost:9000" }
            }
        }

    Returns an empty list when the file is absent, empty, or malformed
    (a warning is printed to stderr in the latter case).
    """
    config_path = _workspace_mcp_config_path(workspace_path)
    if not config_path.exists():
        return []

    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        return _parse_mcp_servers(data)
    except Exception as exc:
        print(
            f"Warning: Failed to parse {config_path}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return []


def save_workspace_mcp_servers(workspace_path: Path, servers: list[tuple[str, str]]) -> None:
    """Write ``(name, url)`` pairs to ``<workspace>/.opendatasci/mcp.json``.

    Replaces the file's whole ``mcpServers`` block with *servers*, creating
    the ``.opendatasci`` directory if needed.
    """
    config_path = _workspace_mcp_config_path(workspace_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    data = {"mcpServers": {name: {"url": url} for name, url in servers}}
    config_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def check_mcp_server(url: str) -> None:
    """Verify an MCP server at *url* is reachable and speaks MCP, raising on failure."""
    _initialize(url)
    _list_tools(url)


def load_mcp_servers(workspace_path: Path) -> list[str]:
    """Read MCP server URLs from ``<workspace>/.opendatasci/mcp.json``.

    See :func:`load_workspace_mcp_servers` for the file format. Returns an
    empty list when the file is absent, empty, or malformed.
    """
    return [url for _, url in load_workspace_mcp_servers(workspace_path)]


def load_named_mcp_servers(config_path: Path) -> list[tuple[str, str]]:
    """Read ``(name, url)`` pairs from an ``mcp.json``-formatted file at *config_path*.

    Unlike :func:`load_mcp_servers`, this reads *config_path* directly (it is
    not resolved relative to a workspace's ``.opendatasci`` directory) and
    raises on a missing or malformed file instead of degrading to an empty
    list — callers driving an interactive picker want to show the user why
    the file couldn't be used.
    """
    if not config_path.is_file():
        raise FileNotFoundError(f"No such file: {config_path}")
    data = json.loads(config_path.read_text(encoding="utf-8"))
    return _parse_mcp_servers(data)


def create_mcp_tools(server_urls: list[str]) -> list[BaseTool]:
    """Query each MCP server URL, fetch its tool manifest, and return wrapped LangChain tools.

    Each MCP server's tools self-register — no ``ToolName`` enum entry is required.
    Servers that fail to connect or respond are skipped with a warning to stderr.

    Args:
        server_urls: List of MCP server base URLs (e.g. ``["http://localhost:8080"]``).

    Returns:
        Flat list of LangChain ``BaseTool`` instances ready to inject into any agent.
    """
    tools: list[BaseTool] = []
    for url in server_urls:
        try:
            _initialize(url)
            tool_defs = _list_tools(url)
        except Exception as exc:
            print(
                f"Warning: Failed to connect to MCP server {url!r}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            continue
        for td in tool_defs:
            try:
                tools.append(_make_mcp_tool(url, td))
            except Exception as exc:
                print(
                    f"Warning: Failed to wrap MCP tool {td.get('name')!r} from {url!r}: {exc}",
                    file=sys.stderr,
                )
    return tools
