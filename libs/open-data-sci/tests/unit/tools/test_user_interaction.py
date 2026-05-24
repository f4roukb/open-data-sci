"""Unit tests for opendatasci.tools.user_interaction."""

from unittest.mock import patch

import pytest

from opendatasci.tools.user_interaction import create_user_interaction_tools

_ARGS = {"question": "Q?", "choice_a": "A", "choice_b": "B", "choice_c": "C"}


def _get_tool():
    return create_user_interaction_tools()[0]


class TestGetUserInteractionToolsStructure:
    def test_returns_one_tool(self) -> None:
        assert len(create_user_interaction_tools()) == 1

    def test_tool_name_is_ask_user_mcq(self) -> None:
        assert create_user_interaction_tools()[0].name == "ask_user_mcq"


class TestAskUserMcq:
    @pytest.mark.asyncio
    async def test_calls_interrupt_with_question_payload(self) -> None:
        tool = _get_tool()
        with patch("opendatasci.tools.user_interaction.interrupt", return_value="A") as mock_intr:
            await tool.ainvoke(_ARGS)
        mock_intr.assert_called_once_with({"question": "Q?", "choices": ["A", "B", "C"]})

    @pytest.mark.asyncio
    async def test_returns_interrupt_result(self) -> None:
        tool = _get_tool()
        with patch("opendatasci.tools.user_interaction.interrupt", return_value="user typed this"):
            result = await tool.ainvoke(_ARGS)
        assert result == "user typed this"

    @pytest.mark.asyncio
    async def test_choices_list_contains_all_three_options(self) -> None:
        tool = _get_tool()
        with patch("opendatasci.tools.user_interaction.interrupt", return_value="B") as mock_intr:
            await tool.ainvoke(
                {"question": "Pick?", "choice_a": "X", "choice_b": "Y", "choice_c": "Z"}
            )
        assert mock_intr.call_args[0][0]["choices"] == ["X", "Y", "Z"]


class TestAskUserMcqCaching:
    @pytest.mark.asyncio
    async def test_cache_hit_skips_interrupt(self) -> None:
        tool = _get_tool()
        with patch("opendatasci.tools.user_interaction.interrupt", return_value="cached"):
            await tool.ainvoke(_ARGS)

        with patch("opendatasci.tools.user_interaction.interrupt") as mock_intr:
            result = await tool.ainvoke(_ARGS)

        mock_intr.assert_not_called()
        assert result == "cached"

    @pytest.mark.asyncio
    async def test_cache_miss_different_question_calls_interrupt_again(self) -> None:
        tool = _get_tool()
        with patch("opendatasci.tools.user_interaction.interrupt", return_value="first"):
            await tool.ainvoke(_ARGS)
        with patch("opendatasci.tools.user_interaction.interrupt", return_value="second") as mock_intr:
            await tool.ainvoke({**_ARGS, "question": "Other?"})
        mock_intr.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_miss_different_choice_calls_interrupt_again(self) -> None:
        tool = _get_tool()
        with patch("opendatasci.tools.user_interaction.interrupt", return_value="ans1"):
            await tool.ainvoke(_ARGS)
        with patch("opendatasci.tools.user_interaction.interrupt", return_value="ans2") as mock_intr:
            await tool.ainvoke({**_ARGS, "choice_a": "Different"})
        mock_intr.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_instances_are_isolated(self) -> None:
        tool1 = _get_tool()
        tool2 = _get_tool()

        with patch("opendatasci.tools.user_interaction.interrupt", return_value="ans1"):
            result1 = await tool1.ainvoke(_ARGS)
        with patch("opendatasci.tools.user_interaction.interrupt", return_value="ans2"):
            result2 = await tool2.ainvoke(_ARGS)

        assert result1 == "ans1"
        assert result2 == "ans2"
