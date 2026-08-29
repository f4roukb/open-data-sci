"""TUI service layer.

``OpenDataSciTuiService``: the single service class used by ``CLIController``.
"""

import logging
from pathlib import Path
from typing import Any, AsyncIterator

from opendatasci.agents.agents import BaseOpenDataSciAgent, Invocation
from opendatasci.sandbox.base import BaseSandbox
from opendatasci.streaming import AgentStreamEvent
from opendatasci.tasks.base import AgentTaskManagerBase

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

    @property
    def task_manager(self) -> AgentTaskManagerBase:
        """The agent's background-task manager (see ``listen_task_updates``)."""
        return self._agent.task_manager

    async def astream(
        self, invocation: Invocation | list[Invocation]
    ) -> AsyncIterator[AgentStreamEvent]:
        """Stream events for *invocation* with token-level output."""
        async for event in self._agent.astream(invocation):
            yield event

    async def resume_with_input(self, answer: str) -> AsyncIterator[AgentStreamEvent]:
        """Resume a pending question/choice prompt with the user's *answer*."""
        async for event in self._agent.resume_with_input(answer):
            yield event

    async def resume_with_approval(self, approved: bool) -> AsyncIterator[AgentStreamEvent]:
        """Resume a pending command-approval prompt with the user's decision."""
        async for event in self._agent.resume_with_approval(approved):
            yield event

    def is_user_input_required(self) -> bool:
        """True iff the agent is paused awaiting the user's answer to a pending question or approval request."""
        return self._agent.is_user_input_required()

    async def reset_session(self) -> None:
        """Reset the execution session and clear agent conversation."""
        self._sandbox.reset()
        await self._agent.clear_chat_history()

    async def clear_context(self) -> None:
        """Clear all agent context: history, summaries, plan, and pending summarizations."""
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
        except OSError:
            logger.exception("Failed to list workspace files at %s", path)
            return []
