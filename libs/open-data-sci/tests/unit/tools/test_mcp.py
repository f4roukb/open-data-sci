"""Unit tests for opendatasci.tools.mcp."""


import json
from io import StringIO
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from opendatasci.tools.mcp import (
    _build_args_model,
    _initialize,
    _jsonrpc,
    _list_tools,
    create_mcp_tools,
    load_mcp_servers,
)

# ---------------------------------------------------------------------------
# _jsonrpc
# ---------------------------------------------------------------------------


class TestJsonrpc:
    def test_includes_jsonrpc_version(self) -> None:
        payload = _jsonrpc("tools/list")
        assert payload["jsonrpc"] == "2.0"

    def test_includes_method(self) -> None:
        payload = _jsonrpc("tools/list")
        assert payload["method"] == "tools/list"

    def test_includes_default_id(self) -> None:
        payload = _jsonrpc("tools/list")
        assert payload["id"] == 1

    def test_custom_id(self) -> None:
        payload = _jsonrpc("tools/list", req_id=5)
        assert payload["id"] == 5

    def test_params_included_when_provided(self) -> None:
        payload = _jsonrpc("initialize", params={"version": "1.0"})
        assert payload["params"] == {"version": "1.0"}

    def test_params_absent_when_none(self) -> None:
        payload = _jsonrpc("tools/list")
        assert "params" not in payload


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
        fields = model.model_fields
        assert "query" in fields
        assert fields["query"].is_required()

    def test_optional_field_defaults_to_none(self) -> None:
        schema: dict[str, Any] = {
            "properties": {"limit": {"type": "integer", "description": "max results"}},
            "required": [],
        }
        model = _build_args_model("test_tool", schema)
        fields = model.model_fields
        assert "limit" in fields
        assert not fields["limit"].is_required()

    def test_type_mapping_string(self) -> None:
        schema: dict[str, Any] = {
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        model = _build_args_model("test_tool", schema)
        instance = model(name="hello")
        assert instance.name == "hello"  # type: ignore[attr-defined]

    def test_type_mapping_integer(self) -> None:
        schema: dict[str, Any] = {
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
        }
        model = _build_args_model("test_tool", schema)
        instance = model(count=42)
        assert instance.count == 42  # type: ignore[attr-defined]

    def test_type_mapping_boolean(self) -> None:
        schema: dict[str, Any] = {
            "properties": {"flag": {"type": "boolean"}},
            "required": ["flag"],
        }
        model = _build_args_model("test_tool", schema)
        instance = model(flag=True)
        assert instance.flag is True  # type: ignore[attr-defined]

    def test_empty_schema_produces_empty_model(self) -> None:
        schema: dict[str, Any] = {"properties": {}, "required": []}
        model = _build_args_model("test_tool", schema)
        instance = model()
        assert instance is not None

    def test_unknown_type_defaults_to_str(self) -> None:
        schema: dict[str, Any] = {
            "properties": {"mystery": {"type": "unknown_type"}},
            "required": ["mystery"],
        }
        model = _build_args_model("test_tool", schema)
        instance = model(mystery="value")
        assert instance.mystery == "value"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# load_mcp_servers
# ---------------------------------------------------------------------------


class TestLoadMcpServerUrls:
    def test_returns_empty_list_when_file_absent(self, tmp_path: Path) -> None:
        result = load_mcp_servers(tmp_path)
        assert result == []

    def test_returns_urls_from_valid_config(self, tmp_path: Path) -> None:
        config_path = tmp_path / ".opendatasci" / "mcp.json"
        config_path.parent.mkdir(parents=True)
        config = {
            "mcpServers": {
                "server-a": {"url": "http://localhost:8080"},
                "server-b": {"url": "http://localhost:9000"},
            }
        }
        config_path.write_text(json.dumps(config))
        result = load_mcp_servers(tmp_path)
        assert "http://localhost:8080" in result
        assert "http://localhost:9000" in result

    def test_returns_empty_list_for_empty_mcp_servers(self, tmp_path: Path) -> None:
        config_path = tmp_path / ".opendatasci" / "mcp.json"
        config_path.parent.mkdir(parents=True)
        config_path.write_text(json.dumps({"mcpServers": {}}))
        result = load_mcp_servers(tmp_path)
        assert result == []

    def test_skips_entries_without_url_key(self, tmp_path: Path) -> None:
        config_path = tmp_path / ".opendatasci" / "mcp.json"
        config_path.parent.mkdir(parents=True)
        config = {"mcpServers": {"no-url": {"host": "localhost"}}}
        config_path.write_text(json.dumps(config))
        result = load_mcp_servers(tmp_path)
        assert result == []

    def test_returns_empty_list_for_malformed_json(self, tmp_path: Path) -> None:
        config_path = tmp_path / ".opendatasci" / "mcp.json"
        config_path.parent.mkdir(parents=True)
        config_path.write_text("not valid json {{{")
        result = load_mcp_servers(tmp_path)
        assert result == []

    def test_prints_warning_for_malformed_json(self, tmp_path: Path) -> None:
        config_path = tmp_path / ".opendatasci" / "mcp.json"
        config_path.parent.mkdir(parents=True)
        config_path.write_text("not valid json")
        captured = StringIO()
        with patch("sys.stderr", captured):
            load_mcp_servers(tmp_path)
        assert "Warning" in captured.getvalue()

    def test_no_mcp_servers_key_returns_empty(self, tmp_path: Path) -> None:
        config_path = tmp_path / ".opendatasci" / "mcp.json"
        config_path.parent.mkdir(parents=True)
        config_path.write_text(json.dumps({"other": {}}))
        result = load_mcp_servers(tmp_path)
        assert result == []


# ---------------------------------------------------------------------------
# create_mcp_tools
# ---------------------------------------------------------------------------


class TestGetMcpTools:
    def test_returns_empty_list_for_no_urls(self) -> None:
        result = create_mcp_tools([])
        assert result == []

    def test_skips_unreachable_server_with_warning(self) -> None:
        captured = StringIO()
        with patch("opendatasci.tools.mcp._initialize", side_effect=ConnectionError("refused")):
            with patch("sys.stderr", captured):
                result = create_mcp_tools(["http://localhost:9999"])
        assert result == []
        assert "Warning" in captured.getvalue()

    def test_wraps_tools_from_reachable_server(self) -> None:
        tool_def = {
            "name": "my_tool",
            "description": "does a thing",
            "inputSchema": {
                "type": "object",
                "properties": {"arg": {"type": "string"}},
                "required": ["arg"],
            },
        }
        with patch("opendatasci.tools.mcp._initialize"):
            with patch("opendatasci.tools.mcp._list_tools", return_value=[tool_def]):
                result = create_mcp_tools(["http://localhost:8080"])
        assert len(result) == 1
        assert result[0].name == "my_tool"

    def test_skips_malformed_tool_definition_with_warning(self) -> None:
        with patch("opendatasci.tools.mcp._initialize"):
            with patch("opendatasci.tools.mcp._list_tools", return_value=[{"no_name": True}]):
                captured = StringIO()
                with patch("sys.stderr", captured):
                    result = create_mcp_tools(["http://localhost:8080"])
        assert result == []

    @pytest.mark.asyncio
    async def test_mcp_tool_call_formats_text_response(self) -> None:
        tool_def = {
            "name": "echo_tool",
            "description": "echoes input",
            "inputSchema": {
                "type": "object",
                "properties": {"msg": {"type": "string"}},
                "required": ["msg"],
            },
        }
        fake_response_data = {"result": {"content": [{"type": "text", "text": "hello from mcp"}]}}
        mock_response = MagicMock()
        mock_response.json.return_value = fake_response_data
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("opendatasci.tools.mcp._initialize"):
            with patch("opendatasci.tools.mcp._list_tools", return_value=[tool_def]):
                tools = create_mcp_tools(["http://localhost:8080"])

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await tools[0].ainvoke({"msg": "hello"})

        assert "hello from mcp" in result

    @pytest.mark.asyncio
    async def test_mcp_tool_call_returns_error_message_on_http_failure(self) -> None:
        tool_def = {
            "name": "fail_tool",
            "description": "fails",
            "inputSchema": {"type": "object", "properties": {}},
        }
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=Exception("connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("opendatasci.tools.mcp._initialize"):
            with patch("opendatasci.tools.mcp._list_tools", return_value=[tool_def]):
                tools = create_mcp_tools(["http://localhost:8080"])

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await tools[0].ainvoke({})

        assert "Error calling MCP tool" in result

    @pytest.mark.asyncio
    async def test_mcp_tool_call_returns_error_on_jsonrpc_error_field(self) -> None:
        tool_def = {
            "name": "err_tool",
            "description": "errors",
            "inputSchema": {"type": "object", "properties": {}},
        }
        fake_response_data = {"error": {"code": -32601, "message": "Method not found"}}
        mock_response = MagicMock()
        mock_response.json.return_value = fake_response_data
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("opendatasci.tools.mcp._initialize"):
            with patch("opendatasci.tools.mcp._list_tools", return_value=[tool_def]):
                tools = create_mcp_tools(["http://localhost:8080"])

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await tools[0].ainvoke({})

        assert "MCP error" in result
        assert "Method not found" in result

    @pytest.mark.asyncio
    async def test_mcp_tool_call_falls_back_to_json_when_no_text_parts(self) -> None:
        """When the MCP response carries no ``type=="text"`` content items, the
        tool must serialise the whole result dict back to JSON so the model can
        still see what the server returned."""
        tool_def = {
            "name": "raw_tool",
            "description": "returns non-text content",
            "inputSchema": {"type": "object", "properties": {}},
        }
        fake_response_data = {
            "result": {
                "content": [{"type": "image", "data": "ignored"}],
                "extra": [1, 2, 3],
            }
        }
        mock_response = MagicMock()
        mock_response.json.return_value = fake_response_data
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("opendatasci.tools.mcp._initialize"):
            with patch("opendatasci.tools.mcp._list_tools", return_value=[tool_def]):
                tools = create_mcp_tools(["http://localhost:8080"])

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await tools[0].ainvoke({})

        parsed = json.loads(result)
        assert parsed == fake_response_data["result"]


# ---------------------------------------------------------------------------
# _initialize — JSON-RPC handshake
# ---------------------------------------------------------------------------


class TestInitialize:
    def _make_sync_client(self, raise_on_second: bool = False) -> tuple[MagicMock, MagicMock]:
        """Build a fake ``httpx.Client`` context manager and the constructor that returns it."""
        response = MagicMock()
        response.raise_for_status = MagicMock()
        client = MagicMock()
        if raise_on_second:
            client.post = MagicMock(side_effect=[response, RuntimeError("boom")])
        else:
            client.post = MagicMock(return_value=response)
        client.__enter__ = MagicMock(return_value=client)
        client.__exit__ = MagicMock(return_value=False)
        ctor = MagicMock(return_value=client)
        return ctor, client

    def test_posts_initialize_payload_to_url(self) -> None:
        ctor, client = self._make_sync_client()
        with patch("opendatasci.tools.mcp.httpx.Client", ctor):
            _initialize("http://srv/mcp")
        first_call = client.post.call_args_list[0]
        assert first_call.args[0] == "http://srv/mcp"
        assert first_call.kwargs["json"]["method"] == "initialize"

    def test_initialize_payload_carries_protocol_version(self) -> None:
        ctor, client = self._make_sync_client()
        with patch("opendatasci.tools.mcp.httpx.Client", ctor):
            _initialize("http://srv/mcp")
        first_call = client.post.call_args_list[0]
        params = first_call.kwargs["json"]["params"]
        assert params["protocolVersion"] == "2024-11-05"
        assert params["clientInfo"]["name"] == "opendatasci"

    def test_sends_initialized_notification_after_handshake(self) -> None:
        ctor, client = self._make_sync_client()
        with patch("opendatasci.tools.mcp.httpx.Client", ctor):
            _initialize("http://srv/mcp")
        assert client.post.call_count == 2
        second_call = client.post.call_args_list[1]
        assert second_call.kwargs["json"]["method"] == "notifications/initialized"

    def test_swallows_notification_failure(self) -> None:
        """Spec says the initialized notification is best-effort — failures must not bubble."""
        ctor, _ = self._make_sync_client(raise_on_second=True)
        with patch("opendatasci.tools.mcp.httpx.Client", ctor):
            _initialize("http://srv/mcp")  # must not raise

    def test_raises_when_initialize_post_fails(self) -> None:
        """The handshake POST itself must propagate errors so callers can drop the server."""
        response = MagicMock()
        response.raise_for_status = MagicMock(side_effect=RuntimeError("502 bad gateway"))
        client = MagicMock()
        client.post = MagicMock(return_value=response)
        client.__enter__ = MagicMock(return_value=client)
        client.__exit__ = MagicMock(return_value=False)
        with patch("opendatasci.tools.mcp.httpx.Client", return_value=client):
            with pytest.raises(RuntimeError, match="502"):
                _initialize("http://srv/mcp")


# ---------------------------------------------------------------------------
# _list_tools — fetch tool manifest
# ---------------------------------------------------------------------------


class TestListTools:
    def _make_client(self, response_json: dict) -> tuple[MagicMock, MagicMock]:
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json = MagicMock(return_value=response_json)
        client = MagicMock()
        client.post = MagicMock(return_value=response)
        client.__enter__ = MagicMock(return_value=client)
        client.__exit__ = MagicMock(return_value=False)
        return MagicMock(return_value=client), client

    def test_posts_tools_list_to_url(self) -> None:
        ctor, client = self._make_client({"result": {"tools": []}})
        with patch("opendatasci.tools.mcp.httpx.Client", ctor):
            _list_tools("http://srv/mcp")
        call = client.post.call_args
        assert call.args[0] == "http://srv/mcp"
        assert call.kwargs["json"]["method"] == "tools/list"

    def test_returns_tool_list_from_response(self) -> None:
        tools_payload = [{"name": "tool_a"}, {"name": "tool_b"}]
        ctor, _ = self._make_client({"result": {"tools": tools_payload}})
        with patch("opendatasci.tools.mcp.httpx.Client", ctor):
            assert _list_tools("http://srv/mcp") == tools_payload

    def test_returns_empty_list_when_result_missing(self) -> None:
        ctor, _ = self._make_client({})
        with patch("opendatasci.tools.mcp.httpx.Client", ctor):
            assert _list_tools("http://srv/mcp") == []

    def test_returns_empty_list_when_tools_key_missing(self) -> None:
        ctor, _ = self._make_client({"result": {}})
        with patch("opendatasci.tools.mcp.httpx.Client", ctor):
            assert _list_tools("http://srv/mcp") == []

    def test_propagates_http_error(self) -> None:
        response = MagicMock()
        response.raise_for_status = MagicMock(side_effect=RuntimeError("500"))
        client = MagicMock()
        client.post = MagicMock(return_value=response)
        client.__enter__ = MagicMock(return_value=client)
        client.__exit__ = MagicMock(return_value=False)
        with patch("opendatasci.tools.mcp.httpx.Client", return_value=client):
            with pytest.raises(RuntimeError, match="500"):
                _list_tools("http://srv/mcp")
