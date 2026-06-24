"""Unit tests for opendatasci.prompts.builders (SystemPromptBuilder).

Conversation memory and the current plan are no longer assembled here — they
are recalled by ChatHistoryBuilder and rendered as standalone HumanMessages
(see tests/unit/agents/test_chat_memory.py). These tests cover prompt
selection, provider caching markers, and skills only.
"""


from functools import partial

import pytest
from langchain_core.messages import SystemMessage

from opendatasci.prompts.prompt_templates import PLAN_MODE_SYSTEM_PROMPT, MAIN_SYSTEM_PROMPT
from opendatasci.prompts.builders import SystemContextBuilder as SystemPromptBuilder
from opendatasci.configs import OpenDataSciConfig
from opendatasci.skills.base import Skill


def _make_active_skills(skill_prompt: str | None) -> list[Skill]:
    return [Skill(name="skill", content=skill_prompt)] if skill_prompt else []


def _extract_text(content: object) -> str:
    """Flatten a SystemMessage content (str or list of dicts) into a single string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
        return " ".join(parts)
    return str(content)


def _all_text(messages: list[SystemMessage]) -> str:
    return "\n".join(_extract_text(m.content) for m in messages)


class TestSystemPromptBuilderBuild:
    def _make(
        self,
        provider: str = "openai",
        is_plan_mode: bool = False,
        skill_prompt: str | None = None,
    ) -> SystemPromptBuilder:
        config = OpenDataSciConfig(provider=provider)
        builder = SystemPromptBuilder(config=config)
        active_skills = _make_active_skills(skill_prompt)
        builder.build = partial(builder.build, active_skills=active_skills, is_plan_mode=is_plan_mode)
        return builder

    # --- Prompt selection ---

    def test_returns_list_of_system_messages(self) -> None:
        builder = self._make()
        result = builder.build()
        assert isinstance(result, list)
        assert all(isinstance(m, SystemMessage) for m in result)

    def test_returns_at_least_one_message(self) -> None:
        builder = self._make()
        assert len(builder.build()) >= 1

    def test_normal_mode_uses_system_prompt(self) -> None:
        builder = self._make()
        assert MAIN_SYSTEM_PROMPT.format(name="Sai") in _all_text(builder.build())

    def test_plan_mode_uses_plan_prompt(self) -> None:
        builder = self._make(is_plan_mode=True)
        assert PLAN_MODE_SYSTEM_PROMPT.format(name="Sai") in _all_text(builder.build())

    # --- Provider-specific caching hints ---

    def test_anthropic_provider_adds_cache_control(self) -> None:
        builder = self._make(provider="anthropic")
        content = builder.build()[0].content
        assert isinstance(content, list)
        assert any("cache_control" in b for b in content if isinstance(b, dict))

    def test_bedrock_provider_adds_cache_point(self) -> None:
        builder = self._make(provider="bedrock")
        content = builder.build()[0].content
        assert isinstance(content, list)
        assert any("cachePoint" in b for b in content if isinstance(b, dict))

    def test_openai_provider_uses_plain_string(self) -> None:
        builder = self._make(provider="openai")
        content = builder.build()[0].content
        assert isinstance(content, str)

    def test_gemini_provider_uses_plain_string(self) -> None:
        builder = self._make(provider="gemini")
        content = builder.build()[0].content
        assert isinstance(content, str)

    def test_ollama_provider_uses_plain_string(self) -> None:
        builder = self._make(provider="ollama")
        content = builder.build()[0].content
        assert isinstance(content, str)

    def test_openai_compatible_server_provider_uses_plain_string(self) -> None:
        builder = self._make(provider="openai_compatible_server")
        content = builder.build()[0].content
        assert isinstance(content, str)

    def test_cached_system_prompt_preserves_text(self) -> None:
        # The wrapped content must always carry the full system prompt,
        # regardless of how the provider chooses to mark up cache breakpoints.
        for provider in (
            "anthropic",
            "bedrock",
            "openai",
            "gemini",
            "ollama",
            "openai_compatible_server",
        ):
            builder = self._make(provider=provider)
            text = _all_text(builder.build())
            assert MAIN_SYSTEM_PROMPT.format(name="Sai") in text, provider

    def test_only_base_message_when_no_skill(self) -> None:
        builder = self._make()
        assert len(builder.build()) == 1

    # --- Skill prompt ---

    def test_includes_skill_prompt(self) -> None:
        builder = self._make(skill_prompt="## My Custom Skill\nDo things well.")
        text = _all_text(builder.build())
        assert "My Custom Skill" in text

    def test_no_skill_message_when_none(self) -> None:
        builder = self._make(skill_prompt=None)
        assert len(builder.build()) == 1

    # --- Combined ---

    def test_base_and_skill_present_when_skill_loaded(self) -> None:
        builder = self._make(skill_prompt="skill instructions")
        messages = builder.build()
        assert len(messages) == 2


def _has_cache_marker(content: object) -> bool:
    """Return True if a SystemMessage content carries a provider cache marker.

    Both supported breakpoint shapes are recognised:
    - Anthropic: a ``{"cache_control": {...}}`` field on a text block.
    - Bedrock:   a ``{"cachePoint": {...}}`` standalone block.
    """
    if not isinstance(content, list):
        return False
    for block in content:
        if not isinstance(block, dict):
            continue
        if "cache_control" in block or "cachePoint" in block:
            return True
    return False


class TestSystemPromptCachingAndOrder:
    """Cache-marker placement and prompt ordering — the contract that lets the
    cached prefix survive across turns without invalidation.

    Invariants enforced here:
    - The base system prompt always carries a cache marker.
    - When a skill is loaded, it sits IMMEDIATELY after the base prompt and
      also carries a cache marker.
    - Total cache markers: exactly 1 with no skill, exactly 2 with a skill.
    """

    def _make(
        self,
        provider: str = "anthropic",
        skill_prompt: str | None = None,
    ) -> SystemPromptBuilder:
        config = OpenDataSciConfig(provider=provider)  # type: ignore[arg-type]
        builder = SystemPromptBuilder(config=config)
        active_skills = _make_active_skills(skill_prompt)
        builder.build = partial(builder.build, active_skills=active_skills)
        return builder

    # --- Cache marker count ---

    @pytest.mark.parametrize("provider", ["anthropic", "bedrock"])
    def test_exactly_one_cache_marker_when_no_skill(self, provider: str) -> None:
        builder = self._make(provider=provider, skill_prompt=None)
        messages = builder.build()
        marked = [m for m in messages if _has_cache_marker(m.content)]
        assert len(marked) == 1

    @pytest.mark.parametrize("provider", ["anthropic", "bedrock"])
    def test_exactly_two_cache_markers_when_skill_loaded(self, provider: str) -> None:
        builder = self._make(provider=provider, skill_prompt="skill body")
        messages = builder.build()
        marked = [m for m in messages if _has_cache_marker(m.content)]
        assert len(marked) == 2

    # --- Cache marker placement ---

    @pytest.mark.parametrize("provider", ["anthropic", "bedrock"])
    def test_base_prompt_always_carries_cache_marker(self, provider: str) -> None:
        builder = self._make(provider=provider)
        messages = builder.build()
        assert _has_cache_marker(messages[0].content)

    @pytest.mark.parametrize("provider", ["anthropic", "bedrock"])
    def test_skill_message_carries_cache_marker(self, provider: str) -> None:
        builder = self._make(provider=provider, skill_prompt="skill body")
        messages = builder.build()
        # The skill message is the only one besides the base that should be cached.
        skill_msgs = [m for m in messages if "skill body" in _extract_text(m.content)]
        assert len(skill_msgs) == 1
        assert _has_cache_marker(skill_msgs[0].content)

    # --- Order: skill immediately after base ---

    def test_skill_is_immediately_after_base_when_loaded(self) -> None:
        builder = self._make(skill_prompt="skill body")
        messages = builder.build()
        # messages[0] is the base prompt; messages[1] must be the skill.
        assert "skill body" in _extract_text(messages[1].content)

    # --- Non-breakpoint providers degrade gracefully ---

    @pytest.mark.parametrize("provider", ["openai", "gemini", "ollama", "openai_compatible_server"])
    def test_breakpoint_providers_skipped_keeps_plain_strings(self, provider: str) -> None:
        # Providers with server-side automatic caching get plain strings for
        # both the base prompt and the skill — there is nothing to mark.
        builder = self._make(provider=provider, skill_prompt="skill body")
        messages = builder.build()
        assert isinstance(messages[0].content, str)
        assert isinstance(messages[1].content, str)
        assert "skill body" in messages[1].content
