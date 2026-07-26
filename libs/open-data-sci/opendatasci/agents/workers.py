"""WorkerAgent: one-shot worker agent for a single delegated subtask.

Deliberately free of any dependency on ``opendatasci.tools`` — unlike
``opendatasci.agents.agents`` (which wires up the main ``Agent`` and its full
tool set, and therefore *does* depend on ``opendatasci.tools``) — so that
tool modules can import :class:`WorkerAgent` directly at module
level. ``opendatasci.tools.tasks`` needs it to build a worker's toolset, and
``opendatasci.tools`` → ``opendatasci.agents.agents`` → ``opendatasci.tools``
would otherwise be a cycle; keeping this class in its own leaf module avoids
that without resorting to a local, function-body import.
"""

import logging
from typing import Any, Callable
from uuid import UUID

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from opendatasci._utils.message_utils import get_message_text_content
from opendatasci.agents.graphs import WorkerGraphFactory
from opendatasci.agents.states import AgentState
from opendatasci.configs import OpenDataSciConfig
from opendatasci.memory.messages import AgentToAgentMessage, MessageOrigin
from opendatasci.models.factory import create_model, with_retry
from opendatasci.prompts.caching import cached_system_prompt
from opendatasci.skills.base import Skill, SkillDomain
from opendatasci.tasks.base import BaseTaskManager, TaskProgressReport, TaskProgressUpdate

logger = logging.getLogger(__name__)

SUBAGENT_TAG: str = "opendatasci:subagent"
WORKER_MAX_STEPS: int = 50

# Signature: (event_type, content, metadata | None) -> None
OnEventCallback = Callable[[str, str, "dict[str, Any] | None"], None]

_ARGS_PREVIEW_LEN = 80


class WorkerAgent:
    """One-shot worker agent that executes a single delegated subtask to completion."""

    def __init__(
        self,
        tools: list[BaseTool],
        config: OpenDataSciConfig | None = None,
        llm: BaseChatModel | None = None,
    ) -> None:
        self._config = config or OpenDataSciConfig()
        _llm = llm if llm is not None else create_model(self._config)
        _llm_with_tools = with_retry(_llm.bind_tools(tools))
        self._current_system_prompt: str = ""

        self._graph = WorkerGraphFactory(
            llm_with_tools=_llm_with_tools,
            tools=tools,
            build_system_context=self._build_system_context,
        ).build()

    def _build_system_context(self, state: AgentState) -> list[SystemMessage]:
        messages: list[SystemMessage] = [
            SystemMessage(
                content=cached_system_prompt(self._current_system_prompt, self._config.provider)  # type: ignore[arg-type]
            )
        ]
        if state.active_skill_domains:
            messages.append(
                SystemMessage(
                    content=cached_system_prompt(
                        state.active_skill_domains[0].content, self._config.provider
                    )  # type: ignore[arg-type]
                )
            )
        for skill in state.active_skills:
            messages.append(
                SystemMessage(
                    content=cached_system_prompt(skill.content, self._config.provider)  # type: ignore[arg-type]
                )
            )
        return messages

    async def ainvoke(
        self,
        task: str,
        system_prompt: str,
        on_event: OnEventCallback | None = None,
        messages_out: "list[Any] | None" = None,
        initial_active_skills: "list[Skill] | None" = None,
        initial_active_skill_domains: "list[SkillDomain] | None" = None,
    ) -> str:
        """Execute *task* to completion and return the final text response."""
        self._current_system_prompt = system_prompt
        initial_state = AgentState(
            messages=[AgentToAgentMessage(content=task, origin=MessageOrigin.AGENT)],
            active_skills=list(initial_active_skills or []),
            active_skill_domains=list(initial_active_skill_domains or []),
        )
        invoke_config: RunnableConfig = {
            "tags": [SUBAGENT_TAG],
            "recursion_limit": WORKER_MAX_STEPS * 2 + 1,
        }

        final_state: dict[str, Any] | None = None

        if on_event is not None:
            async for event in self._graph.astream_events(
                initial_state, version="v2", config=invoke_config
            ):
                kind = event["event"]
                if kind == "on_tool_start":
                    tool_name = event["name"]
                    args = event["data"].get("input") or {}
                    args_preview = str(args)[:_ARGS_PREVIEW_LEN]
                    summary = args.get("summary", "") if isinstance(args, dict) else ""
                    on_event(
                        "task_tool_call",
                        tool_name,
                        {"args_preview": args_preview, "summary": summary},
                    )
                elif kind == "on_tool_end":
                    tool_name = event["name"]
                    output = event["data"].get("output")
                    if isinstance(output, ToolMessage):
                        content = output.content
                    elif isinstance(output, str):
                        content = output
                    else:
                        content = ""
                    is_error = isinstance(content, str) and content.startswith("Error")
                    on_event("task_tool_result", tool_name, {"success": not is_error})
                elif kind == "on_chain_end" and event.get("name") == "LangGraph":
                    final_state = event["data"].get("output")
        else:
            final_state = await self._graph.ainvoke(initial_state, config=invoke_config)

        if messages_out is not None and final_state is not None:
            final_messages = final_state.get("messages", [])
            final_active_skills: list[Skill] = final_state.get("active_skills", [])
            final_active_skill_domains: list[SkillDomain] = final_state.get(
                "active_skill_domains", []
            )
            dummy_state = AgentState(
                messages=[],
                active_skills=final_active_skills,
                active_skill_domains=final_active_skill_domains,
            )
            sys_messages = self._build_system_context(dummy_state)
            messages_out.extend([*sys_messages, *final_messages])

        if final_state is None:
            raise RuntimeError("Worker graph ended without producing output")

        messages = final_state.get("messages", [])
        if not messages:
            raise RuntimeError("Worker graph ended with no messages")

        last = messages[-1]
        return get_message_text_content(last).strip()
