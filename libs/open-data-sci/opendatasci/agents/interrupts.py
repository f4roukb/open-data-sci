"""The "kind" tag every human-input interrupt payload carries."""

from enum import StrEnum


class InterruptKind(StrEnum):
    """Identifies what kind of answer a paused interrupt is waiting for."""

    INPUT_REQUIRED = "input"
    APPROVAL_REQUIRED = "command_approval"
