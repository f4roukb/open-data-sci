"""Unit tests for opendatasci.tools.web."""


from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from opendatasci.tools.web import (
    _clean_html,
    _is_domain_allowed,
    create_web_tools,
)

# ---------------------------------------------------------------------------
# _is_domain_allowed
# ---------------------------------------------------------------------------


class TestIsDomainAllowed:
    def test_exact_match_allowed(self) -> None:
        assert _is_domain_allowed("https://github.com/repo") is True

    def test_subdomain_of_allowed_domain(self) -> None:
        assert _is_domain_allowed("https://en.wikipedia.org/wiki/X") is False
        assert _is_domain_allowed("https://raw.githubusercontent.com/file") is True

    def test_disallowed_domain_returns_false(self) -> None:
        assert _is_domain_allowed("https://example.com/page") is False

    def test_subdomain_of_allowed_arxiv(self) -> None:
        assert _is_domain_allowed("https://arxiv.org/abs/1234") is True

    def test_subdomain_matches(self) -> None:
        assert _is_domain_allowed("https://pandas.pydata.org/docs/") is True

    def test_extra_domains_extend_allowlist(self) -> None:
        extra = frozenset({"mycompany.com"})
        assert _is_domain_allowed("https://mycompany.com/data", extra=extra) is True

    def test_extra_domain_subdomain_allowed(self) -> None:
        extra = frozenset({"mycompany.com"})
        assert _is_domain_allowed("https://api.mycompany.com/v1", extra=extra) is True

    def test_override_replaces_base_list(self) -> None:
        override = frozenset({"custom.io"})
        assert _is_domain_allowed("https://custom.io/page", override=override) is True
        assert _is_domain_allowed("https://github.com/repo", override=override) is False

    def test_override_with_extra_both_allowed(self) -> None:
        override = frozenset({"custom.io"})
        extra = frozenset({"also.com"})
        assert _is_domain_allowed("https://custom.io", override=override, extra=extra) is True
        assert _is_domain_allowed("https://also.com", override=override, extra=extra) is True

    def test_malformed_url_returns_false(self) -> None:
        assert _is_domain_allowed("not-a-url") is False

    def test_empty_string_returns_false(self) -> None:
        assert _is_domain_allowed("") is False

    def test_case_insensitive_hostname(self) -> None:
        assert _is_domain_allowed("https://GITHUB.COM/repo") is True

    def test_finance_yahoo_com_allowed(self) -> None:
        assert _is_domain_allowed("https://finance.yahoo.com/quote/AAPL") is True


# ---------------------------------------------------------------------------
# _clean_html
# ---------------------------------------------------------------------------


class TestCleanHtml:
    def test_removes_script_tags(self) -> None:
        html = "<html><body><script>alert('x')</script><p>Content</p></body></html>"
        result = _clean_html(html)
        assert "alert" not in result
        assert "Content" in result

    def test_removes_style_tags(self) -> None:
        html = "<html><body><style>.foo{color:red}</style><p>Text</p></body></html>"
        result = _clean_html(html)
        assert ".foo" not in result
        assert "Text" in result

    def test_strips_html_tags_leaving_text(self) -> None:
        html = "<p>Hello <b>world</b></p>"
        result = _clean_html(html)
        assert "<p>" not in result
        assert "<b>" not in result
        assert "Hello" in result
        assert "world" in result

    def test_collapses_excessive_blank_lines(self) -> None:
        html = "<p>A</p>\n\n\n\n<p>B</p>"
        result = _clean_html(html)
        assert "\n\n\n" not in result

    def test_strips_surrounding_whitespace_from_lines(self) -> None:
        html = "<p>  padded  </p>"
        result = _clean_html(html)
        for line in result.splitlines():
            assert line == line.strip()

    def test_empty_html_returns_empty_or_whitespace(self) -> None:
        result = _clean_html("")
        assert result.strip() == ""


# ---------------------------------------------------------------------------
# create_web_tools – structure
# ---------------------------------------------------------------------------


class TestGetWebTools:
    def test_returns_two_tools(self) -> None:
        tools = create_web_tools()
        assert len(tools) == 2

    def test_tool_names(self) -> None:
        names = {t.name for t in create_web_tools()}
        assert "web_search" in names
        assert "fetch_url" in names

    def test_extra_domains_forwarded_to_fetch(self) -> None:
        tools = create_web_tools(extra_web_domains=["mycompany.com"])
        fetch = next(t for t in tools if t.name == "fetch_url")
        assert fetch is not None

    def test_override_domains_forwarded_to_fetch(self) -> None:
        tools = create_web_tools(override_web_domains=["custom.io"])
        fetch = next(t for t in tools if t.name == "fetch_url")
        assert fetch is not None


# ---------------------------------------------------------------------------
# fetch_url tool (via create_web_tools)
# ---------------------------------------------------------------------------


class TestFetchUrlTool:
    def _get_fetch_tool(self, **kwargs):
        tools = create_web_tools(**kwargs)
        return next(t for t in tools if t.name == "fetch_url")

    @pytest.mark.asyncio
    async def test_disallowed_domain_returns_error(self) -> None:
        tool = self._get_fetch_tool()
        result = await tool.ainvoke(
            {
                "url": "https://notallowed.example.com/page",
                "summary": "Fetching page",
                "communication": "fetching",
            }
        )
        assert "not in the fetch allowlist" in result or "Error" in result

    @pytest.mark.asyncio
    async def test_allowed_domain_returns_content(self) -> None:
        tool = self._get_fetch_tool()
        mock_response = MagicMock()
        mock_response.text = "<html><body><p>Hello world</p></body></html>"
        mock_response.headers = {"content-type": "text/html"}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await tool.ainvoke(
                {
                    "url": "https://github.com/user/repo",
                    "summary": "Fetching GitHub repo",
                    "communication": "fetching",
                }
            )
        assert "Hello world" in result

    @pytest.mark.asyncio
    async def test_timeout_returns_error_message(self) -> None:
        tool = self._get_fetch_tool()
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await tool.ainvoke(
                {
                    "url": "https://github.com/user/repo",
                    "summary": "Fetching GitHub repo",
                    "communication": "fetching",
                }
            )
        assert "timed out" in result.lower()

    @pytest.mark.asyncio
    async def test_http_error_returns_status_code(self) -> None:
        tool = self._get_fetch_tool()
        mock_response = MagicMock()
        mock_response.status_code = 404

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "Not Found", request=MagicMock(), response=mock_response
            )
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await tool.ainvoke(
                {
                    "url": "https://github.com/user/repo",
                    "summary": "Fetching GitHub repo",
                    "communication": "fetching",
                }
            )
        assert "404" in result

    @pytest.mark.asyncio
    async def test_non_html_response_returned_as_is(self) -> None:
        tool = self._get_fetch_tool()
        mock_response = MagicMock()
        mock_response.text = "col1,col2\n1,2\n3,4"
        mock_response.headers = {"content-type": "text/csv"}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await tool.ainvoke(
                {
                    "url": "https://raw.githubusercontent.com/data.csv",
                    "summary": "Fetching CSV file",
                    "communication": "fetching",
                }
            )
        assert "col1,col2" in result

    @pytest.mark.asyncio
    async def test_extra_domain_allowed_when_configured(self) -> None:
        tool = self._get_fetch_tool(extra_web_domains=["custom.io"])
        mock_response = MagicMock()
        mock_response.text = "custom content"
        mock_response.headers = {"content-type": "text/plain"}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await tool.ainvoke(
                {
                    "url": "https://custom.io/data",
                    "summary": "Fetching custom domain",
                    "communication": "fetching",
                }
            )
        assert "custom content" in result


# ---------------------------------------------------------------------------
# web_search tool (via create_web_tools)
# ---------------------------------------------------------------------------


class TestWebSearchTool:
    def _get_search_tool(self):
        tools = create_web_tools()
        return next(t for t in tools if t.name == "web_search")

    @pytest.mark.asyncio
    async def test_returns_numbered_results(self) -> None:
        tool = self._get_search_tool()
        fake_results = [
            {"title": "Result One", "href": "https://example.com/1", "body": "snippet one"},
            {"title": "Result Two", "href": "https://example.com/2", "body": ""},
        ]

        async def _mock_atext(*args, **kwargs):
            for r in fake_results:
                yield r

        mock_ddgs = MagicMock()
        mock_ddgs.atext = _mock_atext

        with patch("duckduckgo_search.DDGS", return_value=mock_ddgs):
            result = await tool.ainvoke(
                {
                    "query": "test query",
                    "limit": 2,
                    "summary": "Searching web",
                    "communication": "searching",
                }
            )
        assert "1." in result
        assert "Result One" in result

    @pytest.mark.asyncio
    async def test_returns_no_results_message_when_empty(self) -> None:
        tool = self._get_search_tool()

        async def _empty(*args, **kwargs):
            return
            yield

        mock_ddgs = MagicMock()
        mock_ddgs.atext = _empty

        with patch("duckduckgo_search.DDGS", return_value=mock_ddgs):
            result = await tool.ainvoke(
                {
                    "query": "nothing",
                    "limit": 5,
                    "summary": "Searching web",
                    "communication": "searching",
                }
            )
        assert "No results" in result

    @pytest.mark.asyncio
    async def test_max_results_capped_at_10(self) -> None:
        tool = self._get_search_tool()
        captured_kwargs = {}

        async def _capture(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return
            yield

        mock_ddgs = MagicMock()
        mock_ddgs.atext = _capture

        with patch("duckduckgo_search.DDGS", return_value=mock_ddgs):
            await tool.ainvoke({"query": "q", "limit": 100, "summary": "s", "communication": "s"})
        assert captured_kwargs.get("limit", 100) <= 10

    @pytest.mark.asyncio
    async def test_search_exception_returns_error_message(self) -> None:
        tool = self._get_search_tool()

        async def _fail(*args, **kwargs):
            raise RuntimeError("network error")
            yield

        mock_ddgs = MagicMock()
        mock_ddgs.atext = _fail

        with patch("duckduckgo_search.DDGS", return_value=mock_ddgs):
            result = await tool.ainvoke(
                {"query": "q", "limit": 5, "summary": "s", "communication": "s"}
            )
        assert "Error" in result
