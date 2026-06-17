"""Web tools: web_search and fetch_url."""

import re
from collections.abc import Iterable
from functools import lru_cache
from typing import Annotated
from urllib.parse import urlparse

from annotated_types import Ge
from langchain_core.tools import BaseTool, tool

# Domains permitted for fetch_url.  A URL is allowed when its hostname equals
# one of these entries *or* is a subdomain of one (e.g. "en.wikipedia.org"
# matches "wikipedia.org").
_SEARCH_limit: int = 10
_SEARCH_SNIPPET_MAX_CHARS: int = 300

_FETCH_ALLOWED_DOMAINS: frozenset[str] = frozenset(
    {
        # Code & documentation
        "raw.githubusercontent.com",
        "github.com",
        "docs.python.org",
        "pandas.pydata.org",
        "numpy.org",
        "scikit-learn.org",
        "matplotlib.org",
        "scipy.org",
        # ONNX ecosystem
        "onnx.ai",
        "onnxruntime.ai",
        # Research
        "arxiv.org",
        # Finance & open APIs
        "finance.yahoo.com",
        # Competitive data science
        "kaggle.com",
    }
)


_EMPTY_DOMAINS: frozenset[str] = frozenset()


def _is_domain_allowed(
    url: str,
    extra: frozenset[str] = _EMPTY_DOMAINS,
    override: frozenset[str] | None = None,
) -> bool:
    """Return True when *url*'s hostname is in or is a subdomain of the allowed set.

    Args:
        url:      The URL to check.
        extra:    Additional domains unioned with the base set.
        override: When provided, replaces ``_FETCH_ALLOWED_DOMAINS`` entirely
                  before ``extra`` is applied.
    """
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return False
    base = override if override is not None else _FETCH_ALLOWED_DOMAINS
    allowed = base | extra
    return any(host == d or host.endswith("." + d) for d in allowed)


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


@tool
async def web_search(
    query: str, summary: str, communication: str, limit: Annotated[int, Ge(1)] = 10
) -> str:
    """Search the web for resources, documentation, data sources, or reference pages.

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
        limit:         Number of results to return.
    """
    return await _web_search_impl(query, limit)


def _make_fetch_url_tool(
    extra: frozenset[str],
    override: frozenset[str] | None = None,
) -> BaseTool:
    """Return a fetch_url tool bound to the given domain sets."""

    if override is not None:
        allowed_domains = override
    else:
        allowed_domains = _FETCH_ALLOWED_DOMAINS | extra

    @lru_cache(maxsize=16)
    async def _fetch_url_impl(url: str) -> str:
        try:
            import httpx
        except ImportError:
            return "Error: httpx is not installed. Run: pip install httpx"

        if not _is_domain_allowed(url, extra, override):
            host = urlparse(url).hostname or url
            return (
                f"Error: Domain '{host}' is not in the fetch allowlist. "
                "Use web_search to find content from an allowed domain, then fetch that URL."
            )

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

    async def fetch_url(url: str, summary: str, communication: str) -> str:
        return await _fetch_url_impl(url)

    sorted_domains = ", ".join(sorted(allowed_domains))
    fetch_url.__doc__ = (
        f"Fetch the full plain-text content of a URL from an allowed domain.\n\n"
        f"Allowed domains: {sorted_domains}\n\n"
        f"# When to use this tool\n"
        f"- When you have a specific URL from an allowed domain to retrieve.\n"
        f"- To read documentation, papers, or data from a page found via ``web_search``.\n\n"
        f"# When NOT to use this tool\n"
        f"- When the target domain is not in the allowlist — use ``web_search`` instead\n"
        f"  to find useful links that resolve to an allowed domain.\n\n"
        f"Args:\n"
        f"    url:           Full URL to fetch (must be from an allowed domain).\n"
        f'    summary:       3-4 word status label (e.g. "Fetching BLS report").\n'
        f"    communication: Brief message to the user about what you're doing\n"
        f'                   (e.g. "Let me fetch this research paper.").\n'
    )

    return tool(fetch_url)


def create_web_tools(
    extra_web_domains: Iterable[str] = (),
    override_web_domains: Iterable[str] | None = None,
) -> list[BaseTool]:
    """Return the web_search and fetch_url tools (main agent only).

    Args:
        extra_web_domains:   Additional hostnames (or apex domains) to permit
            in ``fetch_url``, beyond the base allowlist.
        override_web_domains: When provided, replaces the built-in allowlist
            entirely.  ``extra_web_domains`` is still applied on top.
    """
    extra = frozenset(d.lower().strip() for d in extra_web_domains)
    override = (
        frozenset(d.lower().strip() for d in override_web_domains)
        if override_web_domains is not None
        else None
    )
    return [web_search, _make_fetch_url_tool(extra, override)]
