"""MCP (Model Context Protocol) integration: connect to HTTP/SSE MCP servers,
discover their tools, and wrap each as an :class:`~opendatasci.tools.base.OpenDataSciBaseTool`.

Only the two remote transports are supported — ``"http"`` (Streamable HTTP)
and ``"sse"`` (the older Server-Sent Events transport some servers still use).
stdio servers (``command``/``args``/``env``) are out of scope: this process
never launches a child process to talk to an MCP server, only HTTP(S) calls.
"""

import asyncio
import hashlib
import json
import re
import sys
from contextlib import asynccontextmanager
from enum import StrEnum
from pathlib import Path
from typing import Any, AsyncIterator, Optional, override

from langchain_core.tools import BaseTool
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import TextContent
from mcp.types import Tool as MCPToolDef
from pydantic import BaseModel, ConfigDict, Field, create_model

from opendatasci.tools.base import OpenDataSciBaseTool

OPENDATASCI_DIRNAME = ".opendatasci"
_MCP_CONFIG_FILE = "mcp.json"

# Timeout for an actual tool call — these can legitimately take a while.
_MCP_CALL_TIMEOUT = 30.0
# Tighter timeout for connectivity checks and tool-discovery passes, which
# run on the manual-add path and at the start of every agent turn and must
# not let one slow/unreachable server stall the whole session.
_MCP_DISCOVERY_TIMEOUT = 8.0

_JSON_SCHEMA_TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


class MCPTransport(StrEnum):
    """The remote MCP transports this client speaks."""

    HTTP = "http"
    SSE = "sse"


class MCPServerSpec(BaseModel):
    """One configured MCP server: how to connect to it and what to call it."""

    model_config = ConfigDict(frozen=True)

    name: str
    url: str
    transport: MCPTransport = MCPTransport.HTTP
    headers: dict[str, str] = Field(default_factory=dict)


def _sanitize_tool_name_part(value: str) -> str:
    """Keep only the characters LLM providers accept in a tool name."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", value) or "server"


def _server_tag(server: MCPServerSpec) -> str:
    """A short, deterministic, collision-resistant tag identifying *server*.

    Server *names* are free text (can collide, differ only by case, contain
    characters no LLM provider accepts in a tool name, ...), so tool names
    are namespaced by a hash of the server's connection identity instead.
    ``hashlib.sha256`` rather than the builtin ``hash()``: ``hash()`` on a
    ``str`` is randomly salted per process (``PYTHONHASHSEED``), so the same
    server would get a different tag every run — sha256 is stable across
    runs and processes, which matters since this tag is what the model
    learns to call the tool by.
    """
    digest = hashlib.sha256(f"{server.transport.value}:{server.url}".encode("utf-8")).hexdigest()
    return digest[:5]


@asynccontextmanager
async def _mcp_session(
    server: MCPServerSpec, *, timeout: float = _MCP_CALL_TIMEOUT
) -> AsyncIterator[ClientSession]:
    """Open a connection to *server*, complete the MCP handshake, and yield a ready session."""
    headers = server.headers or None
    if server.transport is MCPTransport.SSE:
        transport_cm = sse_client(server.url, headers=headers, timeout=timeout)
    else:
        transport_cm = streamable_http_client(server.url, headers=headers, timeout=timeout)
    async with transport_cm as streams:
        read_stream, write_stream = streams[0], streams[1]
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            yield session


async def check_mcp_server(server: MCPServerSpec) -> None:
    """Verify *server* is reachable and speaks MCP, raising on failure."""
    async with _mcp_session(server, timeout=_MCP_DISCOVERY_TIMEOUT) as session:
        await session.list_tools()


def _build_args_model(tool_name: str, input_schema: dict[str, Any]) -> type[BaseModel]:
    """Convert a JSON Schema object into a Pydantic model for the tool's args."""
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

    return create_model(f"_{tool_name}_args", **fields)


class MCPTool(OpenDataSciBaseTool):
    """A single tool exposed by a connected MCP server, discovered dynamically.

    One instance is created per (server, tool) pair on every discovery pass
    (see :func:`discover_mcp_tools`). *name* is namespaced with a short hash
    of the server's connection identity (see :func:`_server_tag`) so two
    servers exposing a same-named tool never collide — server *names* are
    free text and can't be trusted for that. The original, unqualified name
    is kept in *mcp_tool_name* for the actual ``tools/call`` request.
    """

    server: MCPServerSpec
    mcp_tool_name: str

    @override
    async def _arun(self, **kwargs: Any) -> str:
        try:
            async with _mcp_session(self.server) as session:
                result = await session.call_tool(self.mcp_tool_name, kwargs)
        except Exception as exc:
            return (
                f"Error calling MCP tool '{self.mcp_tool_name}' on "
                f"'{self.server.name}': {type(exc).__name__}: {exc}"
            )

        parts = [block.text for block in result.content if isinstance(block, TextContent)]
        if parts:
            text = "\n".join(parts)
        elif result.structuredContent is not None:
            text = json.dumps(result.structuredContent)
        else:
            text = json.dumps([block.model_dump(mode="json") for block in result.content])

        if result.isError:
            return f"MCP error from '{self.mcp_tool_name}': {text or 'unknown error'}"
        return text or "(no output)"


async def _discover_server_tools(server: MCPServerSpec) -> list[BaseTool]:
    """Fetch *server*'s tool manifest and wrap each entry as an :class:`MCPTool`.

    Returns an empty list (with a warning to stderr) if the server can't be
    reached or doesn't respond in time — one bad server never blocks the
    others' tools from loading.
    """
    try:
        async with _mcp_session(server, timeout=_MCP_DISCOVERY_TIMEOUT) as session:
            result = await session.list_tools()
    except Exception as exc:
        print(
            f"Warning: Failed to connect to MCP server '{server.name}' ({server.url}): "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return []

    server_tag = _server_tag(server)
    tools: list[BaseTool] = []
    tool_def: MCPToolDef
    for tool_def in result.tools:
        try:
            args_model = _build_args_model(tool_def.name, tool_def.inputSchema or {})
            tools.append(
                MCPTool(
                    name=f"mcp{server_tag}__{_sanitize_tool_name_part(tool_def.name)}",
                    description=tool_def.description or "",
                    args_schema=args_model,
                    server=server,
                    mcp_tool_name=tool_def.name,
                )
            )
        except Exception as exc:
            print(
                f"Warning: Failed to wrap MCP tool {tool_def.name!r} from '{server.name}': {exc}",
                file=sys.stderr,
            )
    return tools


async def discover_mcp_tools(servers: list[MCPServerSpec]) -> list[BaseTool]:
    """Connect to each of *servers* concurrently and return every tool they expose.

    Meant to be called regularly (once per agent turn), not just at startup —
    a server can enable or disable tools between calls, and this is how the
    agent picks that up. Each server gets a bounded discovery timeout so one
    unreachable server can't stall the rest.
    """
    if not servers:
        return []
    results = await asyncio.gather(*(_discover_server_tools(s) for s in servers))
    return [tool for server_tools in results for tool in server_tools]


def _parse_mcp_servers(data: dict[str, Any]) -> list[MCPServerSpec]:
    """Extract :class:`MCPServerSpec` entries from a decoded ``mcp.json`` document."""
    servers: dict[str, dict[str, Any]] = data.get("mcpServers", {})
    specs: list[MCPServerSpec] = []
    for name, entry in servers.items():
        if not isinstance(entry, dict) or "url" not in entry:
            continue
        try:
            transport = MCPTransport(entry.get("type", entry.get("transport", "http")))
        except ValueError:
            transport = MCPTransport.HTTP
        headers = entry.get("headers")
        specs.append(
            MCPServerSpec(
                name=name,
                url=entry["url"],
                transport=transport,
                headers=headers if isinstance(headers, dict) else {},
            )
        )
    return specs


def _workspace_mcp_config_path(workspace_path: Path) -> Path:
    return workspace_path / OPENDATASCI_DIRNAME / _MCP_CONFIG_FILE


def load_workspace_mcp_servers(workspace_path: Path) -> list[MCPServerSpec]:
    """Read the configured MCP servers from ``<workspace>/.opendatasci/mcp.json``.

    The file format mirrors the Cursor/VS Code ``mcp.json`` convention::

        {
            "mcpServers": {
                "my-server": {
                    "url": "http://localhost:8080",
                    "type": "http",
                    "headers": {"Authorization": "Bearer ..."}
                },
                "another": { "url": "http://localhost:9000", "type": "sse" }
            }
        }

    ``type`` defaults to ``"http"`` and ``headers`` defaults to ``{}`` when
    omitted. Returns an empty list when the file is absent, empty, or
    malformed (a warning is printed to stderr in the latter case).
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


def save_workspace_mcp_servers(workspace_path: Path, servers: list[MCPServerSpec]) -> None:
    """Write *servers* to ``<workspace>/.opendatasci/mcp.json``.

    Replaces the file's whole ``mcpServers`` block, creating the
    ``.opendatasci`` directory if needed.
    """
    config_path = _workspace_mcp_config_path(workspace_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "mcpServers": {
            server.name: {
                "url": server.url,
                "type": server.transport.value,
                **({"headers": server.headers} if server.headers else {}),
            }
            for server in servers
        }
    }
    config_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def load_named_mcp_servers(config_path: Path) -> list[MCPServerSpec]:
    """Read the configured MCP servers from an ``mcp.json``-formatted file at *config_path*.

    Unlike :func:`load_workspace_mcp_servers`, this reads *config_path*
    directly (it is not resolved relative to a workspace's ``.opendatasci``
    directory) and raises on a missing or malformed file instead of
    degrading to an empty list — callers driving an interactive picker want
    to show the user why the file couldn't be used.
    """
    if not config_path.is_file():
        raise FileNotFoundError(f"No such file: {config_path}")
    data = json.loads(config_path.read_text(encoding="utf-8"))
    return _parse_mcp_servers(data)
