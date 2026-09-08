"""Unit tests for opendatasci.human_inputs.human_approval."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from opendatasci.agents.interrupts import InterruptKind
from opendatasci.human_inputs.human_approval import (
    CommandImpactAssessment,
    HumanApprovalBaseManager,
    HumanApprovalManager,
    _CommandImpactAssessment,
)

_MODULE = "opendatasci.human_inputs.human_approval"


def _make_manager(
    description: str = "Lists files.",
    has_negative_impact: bool = False,
    heads_up: str = "",
):
    """Build a HumanApprovalManager whose LLM chain is replaced by a mock."""
    raw = _CommandImpactAssessment(
        description=description,
        has_negative_impact=has_negative_impact,
        heads_up=heads_up,
    )
    mock_structured_llm = MagicMock()
    mock_structured_llm.ainvoke = AsyncMock(return_value=raw)
    mock_base_llm = MagicMock()
    mock_base_llm.with_structured_output.return_value = mock_structured_llm
    with patch(f"{_MODULE}.create_secondary_model", return_value=mock_base_llm):
        manager = HumanApprovalManager(MagicMock())
    return manager, mock_structured_llm


class TestBaseManagerContract:
    def test_abstract_class_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError):
            HumanApprovalBaseManager()  # type: ignore[abstract]

    def test_manager_implements_base_api(self) -> None:
        manager, _ = _make_manager()
        assert isinstance(manager, HumanApprovalBaseManager)


class TestManagerConstruction:
    def test_creates_secondary_model_from_config_once(self) -> None:
        """The secondary model must be used: structured output forces tool_choice,
        which providers such as Anthropic reject when extended thinking is
        enabled (as it is on the primary model)."""
        config = MagicMock()
        with patch(f"{_MODULE}.create_secondary_model") as mock_create:
            HumanApprovalManager(config)
        mock_create.assert_called_once_with(config)

    def test_with_structured_output_uses_private_schema(self) -> None:
        mock_base_llm = MagicMock()
        with patch(f"{_MODULE}.create_secondary_model", return_value=mock_base_llm):
            HumanApprovalManager(MagicMock())
        mock_base_llm.with_structured_output.assert_called_once_with(_CommandImpactAssessment)


class TestAskForCommandApproval:
    @pytest.mark.asyncio
    async def test_interrupt_payload_carries_assessment(self) -> None:
        manager, _ = _make_manager(
            description="Deletes tmp.txt.", has_negative_impact=True, heads_up="File is gone."
        )
        with patch(f"{_MODULE}.interrupt", return_value=True) as mock_intr:
            await manager.ask_for_command_approval("rm tmp.txt")
        mock_intr.assert_called_once_with(
            {
                "kind": InterruptKind.APPROVAL_REQUIRED,
                "command": "rm tmp.txt",
                "description": "Deletes tmp.txt.",
                "heads_up": "File is gone.",
            }
        )

    @pytest.mark.asyncio
    async def test_heads_up_empty_when_no_negative_impact(self) -> None:
        manager, _ = _make_manager(has_negative_impact=False, heads_up="spurious warning")
        with patch(f"{_MODULE}.interrupt", return_value=True) as mock_intr:
            await manager.ask_for_command_approval("ls")
        assert mock_intr.call_args[0][0]["heads_up"] == ""

    @pytest.mark.asyncio
    async def test_yes_answer_returns_true(self) -> None:
        manager, _ = _make_manager()
        with patch(f"{_MODULE}.interrupt", return_value=True):
            assert await manager.ask_for_command_approval("ls") is True

    @pytest.mark.asyncio
    async def test_no_answer_returns_false(self) -> None:
        manager, _ = _make_manager()
        with patch(f"{_MODULE}.interrupt", return_value=False):
            assert await manager.ask_for_command_approval("ls") is False

    @pytest.mark.asyncio
    async def test_llm_receives_system_and_human_messages(self) -> None:
        manager, mock_structured_llm = _make_manager()
        with patch(f"{_MODULE}.interrupt", return_value=True):
            await manager.ask_for_command_approval("ls -la")
        messages = mock_structured_llm.ainvoke.call_args[0][0]
        assert any(isinstance(m, SystemMessage) for m in messages)
        assert any(isinstance(m, HumanMessage) for m in messages)

    @pytest.mark.asyncio
    async def test_command_is_embedded_in_human_message(self) -> None:
        manager, mock_structured_llm = _make_manager()
        with patch(f"{_MODULE}.interrupt", return_value=True):
            await manager.ask_for_command_approval("grep -r secret .")
        human_msg = next(
            m for m in mock_structured_llm.ainvoke.call_args[0][0] if isinstance(m, HumanMessage)
        )
        assert "grep -r secret ." in human_msg.content

    @pytest.mark.asyncio
    async def test_assessment_failure_still_requests_approval(self) -> None:
        """An assessment error must fail closed: the interrupt still fires,
        showing the raw command and a fallback warning."""
        manager, mock_structured_llm = _make_manager()
        mock_structured_llm.ainvoke.side_effect = RuntimeError("LLM unavailable")
        with patch(f"{_MODULE}.interrupt", return_value=True) as mock_intr:
            approved = await manager.ask_for_command_approval("rm tmp.txt")
        assert approved is True
        payload = mock_intr.call_args[0][0]
        assert payload["kind"] == InterruptKind.APPROVAL_REQUIRED
        assert payload["command"] == "rm tmp.txt"
        assert "rm tmp.txt" in payload["description"]
        assert payload["heads_up"]  # fallback warning is always present

    @pytest.mark.asyncio
    async def test_assessment_failure_respects_decline(self) -> None:
        manager, mock_structured_llm = _make_manager()
        mock_structured_llm.ainvoke.side_effect = RuntimeError("LLM unavailable")
        with patch(f"{_MODULE}.interrupt", return_value=False):
            assert await manager.ask_for_command_approval("rm tmp.txt") is False

    @pytest.mark.asyncio
    async def test_manager_is_stateless_across_calls(self) -> None:
        manager, mock_structured_llm = _make_manager()
        with patch(f"{_MODULE}.interrupt", return_value=True):
            await manager.ask_for_command_approval("ls")
            await manager.ask_for_command_approval("ls")
        assert mock_structured_llm.ainvoke.await_count == 2


class TestCommandImpactAssessmentMapping:
    def test_no_impact_maps_heads_up_to_none(self) -> None:
        raw = _CommandImpactAssessment(
            description="Lists files.", has_negative_impact=False, heads_up=""
        )
        assessment = CommandImpactAssessment.from_structured(raw)
        assert assessment.heads_up is None

    def test_impact_maps_heads_up_to_text(self) -> None:
        raw = _CommandImpactAssessment(
            description="Deletes files.", has_negative_impact=True, heads_up="Data loss."
        )
        assessment = CommandImpactAssessment.from_structured(raw)
        assert assessment.heads_up == "Data loss."

    def test_impact_with_blank_heads_up_maps_to_none(self) -> None:
        raw = _CommandImpactAssessment(
            description="Deletes files.", has_negative_impact=True, heads_up="   "
        )
        assessment = CommandImpactAssessment.from_structured(raw)
        assert assessment.heads_up is None

    def test_fields_are_stripped(self) -> None:
        raw = _CommandImpactAssessment(
            description="  Lists files.  ", has_negative_impact=True, heads_up=" Careful. "
        )
        assessment = CommandImpactAssessment.from_structured(raw)
        assert assessment.description == "Lists files."
        assert assessment.heads_up == "Careful."
