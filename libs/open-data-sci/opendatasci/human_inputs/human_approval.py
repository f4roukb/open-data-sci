"""Human-in-the-loop approval for potentially impactful agent actions.

``HumanApprovalManager`` pauses the agent graph before a guarded action runs,
shows the user an LLM-generated plain-language summary of what the agent wants
to execute (plus a heads-up when the action could harm their device or active
work), and resumes with the user's yes/no decision.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from opendatasci.configs import OpenDataSciConfig
from opendatasci.models.factory import create_secondary_model

logger = logging.getLogger(__name__)

APPROVAL_INTERRUPT_KIND = "command_approval"

_APPROVAL_ANSWER_YES = "yes"

_FALLBACK_HEADS_UP = (
    "I tried to assess what this command could do to your device or your active "
    "work, but the check failed temporarily. If you are not sure the command is "
    "safe, play it safe and decline it."
)


class _CommandImpactAssessment(BaseModel):
    """Structured-output schema whose field descriptions prompt the LLM.

    Kept private so prompting stays dissociated from the public
    :class:`CommandImpactAssessment` business object and its documentation.
    """

    description: str = Field(
        description=(
            "One short paragraph, addressed directly to the user, explaining in plain "
            "language what the command will do on their machine and why the agent "
            "wants to run it. No markdown, no jargon."
        )
    )
    has_negative_impact: bool = Field(
        description=(
            "True when running the command could negatively affect the user's device "
            "or active work, in the short or long term (e.g. modifying or deleting "
            "files, consuming significant resources, changing system state, or "
            "leaking data). False for purely read-only, low-cost commands."
        )
    )
    heads_up: str = Field(
        description=(
            "Only when has_negative_impact is true: a concise warning describing the "
            "potential short-term and/or long-term negative impact on the user's "
            "device or active work. Empty string when has_negative_impact is false."
        )
    )


@dataclass(frozen=True)
class CommandImpactAssessment:
    """User-facing summary of a command the agent wants to execute.

    Attributes:
        description: Plain-language paragraph explaining what the command does.
        heads_up: Warning about potential negative impact on the user's device
            or active work, or ``None`` when no negative impact was identified.
    """

    description: str
    heads_up: str | None

    @classmethod
    def from_structured(cls, raw: _CommandImpactAssessment) -> "CommandImpactAssessment":
        heads_up = raw.heads_up.strip() if raw.has_negative_impact else ""
        return cls(description=raw.description.strip(), heads_up=heads_up or None)


_ASSESSMENT_SYSTEM_PROMPT = """\
You are a safety reviewer for an AI data-science agent that runs on the user's \
own machine. The agent wants to execute a CLI command inside the user's active \
workspace directory and needs the user's explicit approval first.

Your job is to brief the user so they can make an informed yes/no decision: \
summarise what the command does, and flag any way accepting it could negatively \
impact their device or active work, whether short-term (e.g. overwriting files, \
heavy resource usage) or long-term (e.g. deleted data, changed configuration). \
Be honest and specific; do not exaggerate risk for harmless read-only commands.\
"""


class HumanApprovalBaseManager(ABC):
    """API for pausing the agent to request the user's approval of an action.

    Concrete implementations decide how the action is summarised and how the
    user's decision is collected. Implementations must be stateless: a single
    instance is created per agent and shared by every tool that needs it.
    """

    @abstractmethod
    async def ask_for_command_approval(self, command: str) -> bool:
        """Pause the agent and ask the user to approve *command*.

        Returns ``True`` when the user approves execution, ``False`` otherwise.
        """


class HumanApprovalManager(HumanApprovalBaseManager):
    """LLM-backed approval manager using LangGraph's ``interrupt()`` mechanism.

    The secondary model generates a :class:`CommandImpactAssessment` for the
    command, then the graph is paused with an interrupt payload carrying the
    assessment; the caller resumes it with the user's answer.  The secondary
    model is used because it has extended thinking disabled: structured output
    forces ``tool_choice``, which providers such as Anthropic reject when
    thinking is enabled (as it is on the primary model).

    If the assessment call fails, approval is still requested — the prompt
    falls back to showing the raw command with a warning instead of silently
    skipping the user's consent.

    Stateless: nothing is cached between calls. Note that LangGraph replays the
    interrupted tool call on resume, so the assessment LLM call runs once more
    when the user answers; the replayed assessment is never shown.
    """

    def __init__(self, config: OpenDataSciConfig) -> None:
        self._llm = create_secondary_model(config).with_structured_output(_CommandImpactAssessment)

    async def ask_for_command_approval(self, command: str) -> bool:
        try:
            assessment = await self._assess(command)
        except Exception:
            logger.warning(
                "Command impact assessment failed; asking for approval with the raw command.",
                exc_info=True,
            )
            assessment = CommandImpactAssessment(
                description=f"The agent wants to run this command in your workspace: {command}",
                heads_up=_FALLBACK_HEADS_UP,
            )
        answer: str = interrupt(
            {
                "kind": APPROVAL_INTERRUPT_KIND,
                "command": command,
                "description": assessment.description,
                "heads_up": assessment.heads_up or "",
            }
        )
        return str(answer).strip().lower() == _APPROVAL_ANSWER_YES

    async def _assess(self, command: str) -> CommandImpactAssessment:
        messages = [
            SystemMessage(content=_ASSESSMENT_SYSTEM_PROMPT),
            HumanMessage(content=f"Command the agent wants to run:\n```\n{command}\n```"),
        ]
        raw: _CommandImpactAssessment = await self._llm.ainvoke(messages)  # type: ignore[assignment]
        return CommandImpactAssessment.from_structured(raw)
