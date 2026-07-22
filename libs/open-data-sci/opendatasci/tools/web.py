"""Web tools: web_search and fetch_url."""

import re
from collections.abc import Coroutine
from functools import lru_cache
from typing import Annotated, Any, Callable, override

from annotated_types import Ge
from langchain_core.tools import BaseTool
from pydantic import BaseModel, PrivateAttr, model_validator

from opendatasci.tools.base import OpenDataSciBaseTool

_SEARCH_limit: int = 10
_SEARCH_SNIPPET_MAX_CHARS: int = 300


def _clean_html(content: str) -> str:
    """Return clean plain text extracted from *content* (HTML)."""
    try:
        from lxml import html as lxml_html

        doc = lxml_html.fromstring(content)
        for el in doc.xpath(
            "//script | //style | //nav | //header | //footer | //aside | //noscript"
        ):
            parent = el.getparent()
            if parent is not None:
                parent.remove(el)
        text: str = doc.text_content()
    except Exception:
        text = re.sub(
            r"<\s*script\b[^>]*>.*?<\s*/\s*script\b[^>]*>",
            "",
            content,
            flags=re.DOTALL | re.IGNORECASE,
        )
        text = re.sub(
            r"<\s*style\b[^>]*>.*?<\s*/\s*style\b[^>]*>", "", text, flags=re.DOTALL | re.IGNORECASE
        )
        text = re.sub(r"<[^>]+>", "", text)

    lines = [ln.strip() for ln in text.splitlines()]
    text = "\n".join(ln for ln in lines if ln)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


@lru_cache(maxsize=16)
async def _web_search_impl(query: str, limit: int) -> str:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return "Error: duckduckgo-search is not installed. Run: pip install duckduckgo-search"

    n = max(1, min(int(limit), _SEARCH_limit))
    try:
        results = [r async for r in DDGS().atext(query, limit=n)]  # type: ignore[attr-defined]
    except Exception as exc:
        return f"Error performing web search: {type(exc).__name__}: {exc}"

    if not results:
        return "No results found."

    lines = []
    for i, r in enumerate(results, 1):
        line = f"{i}. {r['title']} — {r['href']}"
        body = (r.get("body") or "").strip()[:_SEARCH_SNIPPET_MAX_CHARS]
        if body:
            line += f"\n   {body}"
        lines.append(line)
    return "\n".join(lines)


def _build_fetch_url_impl() -> Callable[[str], Coroutine[Any, Any, str]]:
    """Return a URL-fetching coroutine function with its own cache.

    The cache is scoped to this closure (fresh per tool instance) rather than
    a shared module-level cache, so repeated fetches of the same URL across
    unrelated tool instances (e.g. in tests) don't reuse an already-awaited
    coroutine.
    """

    @lru_cache(maxsize=16)
    async def _fetch_url_impl(url: str) -> str:
        try:
            import httpx
        except ImportError:
            return "Error: httpx is not installed. Run: pip install httpx"

        if not (url.startswith("http://") or url.startswith("https://")):
            return f"Error: '{url}' is not a valid http(s) URL."

        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=20,
                headers={"User-Agent": "Mozilla/5.0 (compatible; OpenDataSci/1.0)"},
            ) as client:
                response = await client.get(url)
            response.raise_for_status()
        except httpx.TimeoutException:
            return "Error: Request timed out after 20 seconds."
        except httpx.HTTPStatusError as exc:
            return f"Error: HTTP {exc.response.status_code} fetching {url}"
        except Exception as exc:
            return f"Error fetching URL: {type(exc).__name__}: {exc}"

        content_type = response.headers.get("content-type", "").lower()
        if "html" in content_type:
            return _clean_html(response.text)
        return response.text

    return _fetch_url_impl


class WebSearchTool(OpenDataSciBaseTool):
    """Search the web for resources, documentation, data sources, or reference pages."""

    class CallArgs(BaseModel):
        query: str
        summary: str
        communication: str
        limit: Annotated[int, Ge(1)] = 10

    name: str = "web_search"
    description: str = """\
Search the web for resources, documentation, data sources, or reference pages.

Returns titles, URLs, and short snippets. Follow up with ``fetch_url`` to retrieve full content.

# When to use this tool
- To discover data sources, APIs, documentation, or research papers.
- When you don't know the exact URL of the resource you need.

# How to use this tool
- Keep queries specific: include key terms rather than full sentences.
- Follow up with ``fetch_url`` on the most relevant result to get full content.

Args:
    query:         Search query (natural language or keywords).
    summary:       3-4 word status label (e.g. "Searching BLS data").
    communication: Brief message to the user about what you're doing
                   (e.g. "Let me search for data sources that could be useful for this task.").
    limit:         Number of results to return.\
"""
    args_schema: type[BaseModel] = CallArgs

    @override
    async def _arun(
        self, query: str, summary: str, communication: str, limit: int = 10, **kwargs: Any
    ) -> str:
        return await _web_search_impl(query, limit)


class FetchUrlTool(OpenDataSciBaseTool):
    """Fetch the full plain-text content of a URL."""

    class CallArgs(BaseModel):
        url: str
        summary: str
        communication: str

    name: str = "fetch_url"
    description: str = """\
Fetch the full plain-text content of a URL.

# When to use this tool
- When you have a specific URL to retrieve.
- To read documentation, papers, or data from a page found via ``web_search``.

Args:
    url:           Full URL to fetch.
    summary:       3-4 word status label (e.g. "Fetching BLS report").
    communication: Brief message to the user about what you're doing
                   (e.g. "Let me fetch this research paper.").\
"""
    args_schema: type[BaseModel] = CallArgs

    _fetch_impl: Callable[[str], Coroutine[Any, Any, str]] = PrivateAttr()

    @model_validator(mode="after")
    def _setup(self) -> "FetchUrlTool":
        self._fetch_impl = _build_fetch_url_impl()
        return self

    @override
    async def _arun(self, url: str, summary: str, communication: str, **kwargs: Any) -> str:
        return await self._fetch_impl(url)


def create_web_tools() -> list[BaseTool]:
    """Return the web_search and fetch_url tools (main agent only)."""
    return [WebSearchTool(), FetchUrlTool()]
