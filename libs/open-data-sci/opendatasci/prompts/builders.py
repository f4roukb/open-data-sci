from __future__ import annotations

from langchain_core.messages import SystemMessage

from opendatasci.configs import OpenDataSciConfig
from opendatasci.prompts.caching import cached_system_prompt
from opendatasci.prompts.prompt_templates import (
    MAIN_SYSTEM_PROMPT,
    PLAN_MODE_SYSTEM_PROMPT,
    SELF_REVIEW_MODE_SYSTEM_PROMPT,
)
from opendatasci.skills.base import Skill

__all__ = [
    "SystemContextBuilder",
]


class SystemContextBuilder:
    """Assembles the system prompt for each conversation turn.

    Emits system messages in this order:
    1. Base prompt (main, plan, or self-review depending on mode) — cached.
    2. One message per active skill — each cached.

    Conversation recall (rolling turn summaries) and the current plan are not
    part of the system prompt — ``ChatHistoryBuilder`` renders them as
    standalone ``HumanMessage``s instead (see
    :func:`opendatasci.agents.chat_memory.build_chat_recap_messages` and
    :func:`opendatasci.agents.chat_memory.build_plan_message`).
    """

    def __init__(self, config: OpenDataSciConfig) -> None:
        self._config = config

    def build(
        self,
        active_skills: list[Skill] | None = None,
        is_plan_mode: bool = False,
        is_self_review_mode: bool = False,
    ) -> list[SystemMessage]:
        """Build and return the system prompt messages for the current agent state."""

        if is_plan_mode:
            prompt = PLAN_MODE_SYSTEM_PROMPT
        elif is_self_review_mode:
            prompt = SELF_REVIEW_MODE_SYSTEM_PROMPT
        else:
            prompt = MAIN_SYSTEM_PROMPT

        # Stable prefix — carries the cache breakpoint(s). The skill, when
        # loaded, sits immediately after the base prompt so the cached prefix
        # extends through it without invalidation on subsequent turns.
        base_msg = SystemMessage(
            content=cached_system_prompt(
                prompt.format(name=self._config.name), self._config.provider
            )  # type: ignore[arg-type]
        )
        messages: list[SystemMessage] = [base_msg]

        for skill in active_skills or []:
            messages.append(
                SystemMessage(
                    content=cached_system_prompt(skill.content, self._config.provider)  # type: ignore[arg-type]
                )
            )

        return messages
