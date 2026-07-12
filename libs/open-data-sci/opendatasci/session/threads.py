"""Conversation thread representation."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class SessionThread(BaseModel):
    """One conversation thread in the graph checkpointer.

    A session accumulates threads over time: clearing the conversation
    creates a new thread, abandoning the previous one.
    """

    thread_id: UUID
    created_at: datetime
