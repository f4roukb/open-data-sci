from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import SystemMessage

from opendatasci.configs import OpenDataSciConfig
from opendatasci.prompts.caching import cached_system_prompt
from opendatasci.prompts.message_templates import PLAN_SYSTEM_MESSAGE_TEMPLATE
from opendatasci.prompts.prompt_templates import (
    MAIN_SYSTEM_PROMPT,
    PLAN_MODE_SYSTEM_PROMPT,
    SELF_REVIEW_MODE_SYSTEM_PROMPT,
)
from opendatasci.skills.base import Skill

__all__ = [
    "SystemContextBuilder",
]

if TYPE_CHECKING:
    from opendatasci.context.base import BaseContextStore


class SystemContextBuilder:
    """Assembles the system prompt for each conversation turn.

    Emits system messages in this order:
    1. Base prompt (main, plan, or self-review depending on mode) — cached.
    2. One message per active skill — each cached.
    3. Plan tail (dynamic, not cached) — when a plan exists for the session.
    4. Memory tail (dynamic, not cached) — when recalled context is provided.
    """

    def __init__(
        self,
        config: OpenDataSciConfig,
        context_store: "BaseContextStore",
        session_id: str,
    ) -> None:
        self._config = config
        self._context_store = context_store
        self._session_id = session_id

    def build(
        self,
        active_skills: list[Skill] | None = None,
        is_plan_mode: bool = False,
        is_self_review_mode: bool = False,
        memory_text: str | None = None,
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

        # Dynamic tails — change between turns, never wrapped with cache markers.
        if self._context_store and (plan := self._context_store.current_plan(self._session_id)):
            messages.append(SystemMessage(content=PLAN_SYSTEM_MESSAGE_TEMPLATE.format(plan=plan)))

        if memory_text:
            messages.append(SystemMessage(content=memory_text))

        return messages
