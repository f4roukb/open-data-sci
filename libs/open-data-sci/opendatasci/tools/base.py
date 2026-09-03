"""Base class for OpenDataSci tools."""

from abc import abstractmethod
from typing import Any, override

from langchain_core.tools import BaseTool

from opendatasci._utils.casing_utils import camel_to_snake_keys


class OpenDataSciBaseTool(BaseTool):
    """Base class for async-only OpenDataSci tools.

    Subclasses set the usual :class:`BaseTool` fields (``name``,
    ``description``, ``args_schema``) as class attributes, declare any bound
    dependencies (sandbox, skill store, config, ...) as additional pydantic
    fields, and implement :meth:`_arun` with the tool's actual behavior.
    Every tool in this codebase is async-only; ``_run`` is intentionally not
    supported.
    """

    @override
    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(f"'{self.name}' only supports async execution (_arun).")

    @abstractmethod
    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the tool."""
        ...

    @override
    def _parse_input(
        self, tool_input: str | dict[str, Any], tool_call_id: str | None
    ) -> str | dict[str, Any]:
        """Normalize camelCase argument keys before validation.

        Some models emit camelCase keys for multi-word snake_case parameters
        (e.g. ``requestApproval`` for ``request_approval``) despite the tool
        schema advertising the snake_case name. Left uncorrected, LangChain's
        ``BaseTool._parse_input`` silently drops the value: it re-derives the
        validated dict by intersecting field names against the *original*
        input's keys, so a value that only validated via a camelCase alias
        still gets filtered out. Normalizing keys up front sidesteps that
        entirely.
        """
        if isinstance(tool_input, dict):
            tool_input = camel_to_snake_keys(tool_input)
        return super()._parse_input(tool_input, tool_call_id)
