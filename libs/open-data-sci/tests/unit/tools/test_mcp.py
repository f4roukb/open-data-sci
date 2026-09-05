"""Unit tests for opendatasci.tools.mcp."""

import json
from contextlib import asynccontextmanager
from io import StringIO
from pathlib import Path
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.types import TextContent

from opendatasci.tools.mcp import (
    MCPServerSpec,
    MCPTool,
    MCPTransport,
    _build_args_model,
    _parse_mcp_servers,
    _server_tag,
    check_mcp_server,
    discover_mcp_tools,
    load_named_mcp_servers,
    load_workspace_mcp_servers,
    save_workspace_mcp_servers,
)

_HTTP_SERVER = MCPServerSpec(name="server-a", url="http://localhost:8080")


def _fake_session(**overrides: Any) -> MagicMock:
    session = MagicMock()
    session.list_tools = AsyncMock(return_value=MagicMock(tools=[]))
    session.call_tool = AsyncMock()
    session.initialize = AsyncMock()
    for key, value in overrides.items():
        setattr(session, key, value)
    return session


def _patched_session(session: MagicMock):
    @asynccontextmanager
    async def _cm(server: MCPServerSpec, *, timeout: float = 0.0) -> AsyncIterator[MagicMock]:
        yield session

    return patch("opendatasci.tools.mcp._mcp_session", _cm)


# ---------------------------------------------------------------------------
# MCPServerSpec / _parse_mcp_servers
# ---------------------------------------------------------------------------


class TestParseMcpServers:
    def test_defaults_to_http_transport(self) -> None:
        data = {"mcpServers": {"a": {"url": "http://localhost:8080"}}}
        specs = _parse_mcp_servers(data)
        assert specs == [MCPServerSpec(name="a", url="http://localhost:8080")]
        assert specs[0].transport is MCPTransport.HTTP

    def test_reads_sse_transport(self) -> None:
        data = {"mcpServers": {"a": {"url": "http://localhost:8080", "type": "sse"}}}
        specs = _parse_mcp_servers(data)
        assert specs[0].transport is MCPTransport.SSE

    def test_unknown_transport_falls_back_to_http(self) -> None:
        data = {"mcpServers": {"a": {"url": "http://localhost:8080", "type": "stdio"}}}
        specs = _parse_mcp_servers(data)
        assert specs[0].transport is MCPTransport.HTTP

    def test_reads_headers(self) -> None:
        data = {
            "mcpServers": {
                "a": {"url": "http://localhost:8080", "headers": {"Authorization": "Bearer x"}}
            }
        }
        specs = _parse_mcp_servers(data)
        assert specs[0].headers == {"Authorization": "Bearer x"}

    def test_missing_headers_defaults_to_empty_dict(self) -> None:
        data = {"mcpServers": {"a": {"url": "http://localhost:8080"}}}
        specs = _parse_mcp_servers(data)
        assert specs[0].headers == {}

    def test_skips_entries_without_url_key(self) -> None:
        data = {"mcpServers": {"no-url": {"host": "localhost"}}}
        assert _parse_mcp_servers(data) == []

    def test_no_mcp_servers_key_returns_empty(self) -> None:
        assert _parse_mcp_servers({"other": {}}) == []


# ---------------------------------------------------------------------------
# load_workspace_mcp_servers / save_workspace_mcp_servers
# ---------------------------------------------------------------------------


class TestLoadWorkspaceMcpServers:
    def test_returns_empty_list_when_file_absent(self, tmp_path: Path) -> None:
        assert load_workspace_mcp_servers(tmp_path) == []

    def test_returns_specs_from_valid_config(self, tmp_path: Path) -> None:
        config_path = tmp_path / ".opendatasci" / "mcp.json"
        config_path.parent.mkdir(parents=True)
        config = {
            "mcpServers": {
                "server-a": {"url": "http://localhost:8080"},
                "server-b": {"url": "http://localhost:9000", "type": "sse"},
            }
        }
        config_path.write_text(json.dumps(config))
        result = load_workspace_mcp_servers(tmp_path)
        assert MCPServerSpec(name="server-a", url="http://localhost:8080") in result
        assert (
            MCPServerSpec(name="server-b", url="http://localhost:9000", transport=MCPTransport.SSE)
            in result
        )

    def test_returns_empty_list_for_malformed_json(self, tmp_path: Path) -> None:
        config_path = tmp_path / ".opendatasci" / "mcp.json"
        config_path.parent.mkdir(parents=True)
        config_path.write_text("not valid json {{{")
        assert load_workspace_mcp_servers(tmp_path) == []

    def test_prints_warning_for_malformed_json(self, tmp_path: Path) -> None:
        config_path = tmp_path / ".opendatasci" / "mcp.json"
        config_path.parent.mkdir(parents=True)
        config_path.write_text("not valid json")
        captured = StringIO()
        with patch("sys.stderr", captured):
            load_workspace_mcp_servers(tmp_path)
        assert "Warning" in captured.getvalue()


class TestSaveWorkspaceMcpServers:
    def test_writes_url_type_and_headers(self, tmp_path: Path) -> None:
        servers = [
            MCPServerSpec(
                name="server-a",
                url="http://localhost:8080",
                transport=MCPTransport.SSE,
                headers={"X-Api-Key": "secret"},
            )
        ]
        save_workspace_mcp_servers(tmp_path, servers)
        data = json.loads((tmp_path / ".opendatasci" / "mcp.json").read_text())
        entry = data["mcpServers"]["server-a"]
        assert entry["url"] == "http://localhost:8080"
        assert entry["type"] == "sse"
        assert entry["headers"] == {"X-Api-Key": "secret"}

    def test_omits_headers_key_when_empty(self, tmp_path: Path) -> None:
        save_workspace_mcp_servers(tmp_path, [MCPServerSpec(name="a", url="http://x")])
        data = json.loads((tmp_path / ".opendatasci" / "mcp.json").read_text())
        assert "headers" not in data["mcpServers"]["a"]

    def test_round_trips_through_load(self, tmp_path: Path) -> None:
        servers = [MCPServerSpec(name="a", url="http://x", transport=MCPTransport.SSE)]
        save_workspace_mcp_servers(tmp_path, servers)
        assert load_workspace_mcp_servers(tmp_path) == servers


class TestLoadNamedMcpServers:
    def test_raises_for_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_named_mcp_servers(tmp_path / "nope.json")

    def test_reads_specs(self, tmp_path: Path) -> None:
        config_path = tmp_path / "mcp.json"
        config_path.write_text(json.dumps({"mcpServers": {"a": {"url": "http://x"}}}))
        assert load_named_mcp_servers(config_path) == [MCPServerSpec(name="a", url="http://x")]


# ---------------------------------------------------------------------------
# _build_args_model
# ---------------------------------------------------------------------------


class TestBuildArgsModel:
    def test_required_field_has_no_default(self) -> None:
        schema: dict[str, Any] = {
            "properties": {"query": {"type": "string", "description": "search query"}},
            "required": ["query"],
        }
        model = _build_args_model("test_tool", schema)
        assert model.model_fields["query"].is_required()

    def test_optional_field_defaults_to_none(self) -> None:
        schema: dict[str, Any] = {
            "properties": {"limit": {"type": "integer", "description": "max results"}},
            "required": [],
        }
        model = _build_args_model("test_tool", schema)
        assert not model.model_fields["limit"].is_required()

    def test_empty_schema_produces_empty_model(self) -> None:
        model = _build_args_model("test_tool", {"properties": {}, "required": []})
        assert model() is not None


# ---------------------------------------------------------------------------
# check_mcp_server
# ---------------------------------------------------------------------------


class TestCheckMcpServer:
    async def test_raises_when_unreachable(self) -> None:
        @asynccontextmanager
        async def _boom(server: MCPServerSpec, *, timeout: float = 0.0) -> AsyncIterator[Any]:
            raise ConnectionError("refused")
            yield  # pragma: no cover

        with patch("opendatasci.tools.mcp._mcp_session", _boom):
            with pytest.raises(ConnectionError):
                await check_mcp_server(_HTTP_SERVER)

    async def test_lists_tools_on_success(self) -> None:
        session = _fake_session()
        with _patched_session(session):
            await check_mcp_server(_HTTP_SERVER)
        session.list_tools.assert_awaited_once()


# ---------------------------------------------------------------------------
# discover_mcp_tools
# ---------------------------------------------------------------------------


class TestDiscoverMcpTools:
    async def test_returns_empty_list_for_no_servers(self) -> None:
        assert await discover_mcp_tools([]) == []

    async def test_skips_unreachable_server_with_warning(self) -> None:
        @asynccontextmanager
        async def _boom(server: MCPServerSpec, *, timeout: float = 0.0) -> AsyncIterator[Any]:
            raise ConnectionError("refused")
            yield  # pragma: no cover

        captured = StringIO()
        with patch("opendatasci.tools.mcp._mcp_session", _boom), patch("sys.stderr", captured):
            result = await discover_mcp_tools([_HTTP_SERVER])
        assert result == []
        assert "Warning" in captured.getvalue()

    async def test_wraps_tools_from_reachable_server(self) -> None:
        tool_def = MagicMock(
            description="does a thing",
            inputSchema={
                "type": "object",
                "properties": {"arg": {"type": "string"}},
                "required": ["arg"],
            },
        )
        tool_def.name = "my_tool"
        session = _fake_session(list_tools=AsyncMock(return_value=MagicMock(tools=[tool_def])))
        with _patched_session(session):
            result = await discover_mcp_tools([_HTTP_SERVER])
        assert len(result) == 1
        assert isinstance(result[0], MCPTool)
        assert result[0].name == f"mcp{_server_tag(_HTTP_SERVER)}__my_tool"
        assert result[0].mcp_tool_name == "my_tool"

    async def test_namespaces_tools_by_server_to_avoid_collisions(self) -> None:
        def _tool_def(name: str) -> MagicMock:
            td = MagicMock(description="", inputSchema={})
            td.name = name
            return td

        server_a = MCPServerSpec(name="a", url="http://a")
        server_b = MCPServerSpec(name="b", url="http://b")

        sessions = {
            "http://a": _fake_session(
                list_tools=AsyncMock(return_value=MagicMock(tools=[_tool_def("shared_tool")]))
            ),
            "http://b": _fake_session(
                list_tools=AsyncMock(return_value=MagicMock(tools=[_tool_def("shared_tool")]))
            ),
        }

        @asynccontextmanager
        async def _cm(server: MCPServerSpec, *, timeout: float = 0.0) -> AsyncIterator[MagicMock]:
            yield sessions[server.url]

        with patch("opendatasci.tools.mcp._mcp_session", _cm):
            result = await discover_mcp_tools([server_a, server_b])
        names = {t.name for t in result}
        assert names == {
            f"mcp{_server_tag(server_a)}__shared_tool",
            f"mcp{_server_tag(server_b)}__shared_tool",
        }
        assert len(names) == 2  # different servers must get different tags

    async def test_skips_malformed_tool_definition_with_warning(self) -> None:
        bad_def = MagicMock(description="", inputSchema="not-a-dict")
        bad_def.name = "bad_tool"
        session = _fake_session(list_tools=AsyncMock(return_value=MagicMock(tools=[bad_def])))
        captured = StringIO()
        with _patched_session(session), patch("sys.stderr", captured):
            result = await discover_mcp_tools([_HTTP_SERVER])
        assert result == []
        assert "Warning" in captured.getvalue()


# ---------------------------------------------------------------------------
# MCPTool._arun
# ---------------------------------------------------------------------------


class TestMCPToolArun:
    def _tool(self) -> MCPTool:
        return MCPTool(
            name=f"mcp{_server_tag(_HTTP_SERVER)}__echo",
            description="echoes",
            args_schema=_build_args_model("echo", {"properties": {}, "required": []}),
            server=_HTTP_SERVER,
            mcp_tool_name="echo",
        )

    async def test_formats_text_response(self) -> None:
        content = [TextContent(type="text", text="hello from mcp")]
        result_obj = MagicMock(content=content, isError=False, structuredContent=None)
        session = _fake_session(call_tool=AsyncMock(return_value=result_obj))
        with _patched_session(session):
            result = await self._tool()._arun(msg="hi")
        assert result == "hello from mcp"

    async def test_falls_back_to_structured_content_when_no_text_parts(self) -> None:
        result_obj = MagicMock(content=[], isError=False, structuredContent={"a": 1})
        session = _fake_session(call_tool=AsyncMock(return_value=result_obj))
        with _patched_session(session):
            result = await self._tool()._arun()
        assert json.loads(result) == {"a": 1}

    async def test_reports_mcp_error(self) -> None:
        content = [TextContent(type="text", text="bad args")]
        result_obj = MagicMock(content=content, isError=True, structuredContent=None)
        session = _fake_session(call_tool=AsyncMock(return_value=result_obj))
        with _patched_session(session):
            result = await self._tool()._arun()
        assert "MCP error" in result
        assert "bad args" in result

    async def test_connection_failure_returns_error_message(self) -> None:
        @asynccontextmanager
        async def _boom(server: MCPServerSpec, *, timeout: float = 0.0) -> AsyncIterator[Any]:
            raise ConnectionError("refused")
            yield  # pragma: no cover

        with patch("opendatasci.tools.mcp._mcp_session", _boom):
            result = await self._tool()._arun()
        assert "Error calling MCP tool" in result
