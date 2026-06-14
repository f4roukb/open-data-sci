"""TUI service layer.

``OpenDataSciTuiService``: the single service class used by ``CLIController``.
"""

import logging
from pathlib import Path
from typing import Any, AsyncIterator

from opendatasci.agents.agents import BaseOpenDataSciAgent
from opendatasci.sandbox.base import BaseSandbox
from opendatasci.streaming import AgentStreamEvent

logger = logging.getLogger(__name__)


class OpenDataSciTuiService:
    """Service layer for the OpenDataSci TUI.

    Owns the agent and sandbox for the lifetime of a terminal session.
    Create a new instance for each file or workspace loaded by the TUI.
    """

    def __init__(
        self,
        agent: BaseOpenDataSciAgent,
        sandbox: BaseSandbox,
        workspace_path: Path | None = None,
    ) -> None:
        self._agent = agent
        self._sandbox = sandbox
        self._workspace_path = workspace_path

    async def __aenter__(self) -> "OpenDataSciTuiService":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    async def close(self) -> None:
        """Release sandbox resources (e.g. stop Docker containers)."""
        await self._sandbox.close()

    async def astream(self, query: str) -> AsyncIterator[AgentStreamEvent]:
        """Stream events for *query* with token-level output."""
        async for event in self._agent.astream(query):
            yield event

    async def reset_session(self) -> None:
        """Reset the execution session and clear agent conversation."""
        self._sandbox.reset()
        await self._agent.clear_chat_history()

    async def clear_context(self) -> None:
        """Clear all agent context: conversation history and memory summaries."""
        await self._agent.clear_chat_history()

    async def rewind_turn(self) -> None:
        """Remove the last turn from the conversation history."""
        await self._agent.rewind_turn()

    async def compact_chat_history(self) -> str:
        """Compact the conversation history and return the summary."""
        return await self._agent.compact_chat_history()

    def get_workspace_files(self) -> list[str]:
        """Return names of files/dirs visible in the workspace, relative to its root.

        Used by the /ls-workspace command.
        """
        if self._workspace_path is None:
            return []
        path = self._workspace_path
        try:
            entries = sorted(path.iterdir(), key=lambda f: (f.is_dir(), f.name.lower()))
            return [e.name + ("/" if e.is_dir() else "") for e in entries]
        except Exception:
            return []
