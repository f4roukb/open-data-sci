"""Helpers for rendering background-task updates as model-facing messages."""

from opendatasci.memory.messages import TaskMessage
from opendatasci.tasks.base import BackgroundTaskUpdate


def merge_task_updates(updates: list[BackgroundTaskUpdate]) -> TaskMessage:
    """Render a same-``task_id`` group of updates as a single :class:`TaskMessage`.

    A singleton group renders identically to that update's own
    :meth:`BackgroundTaskUpdate.to_message`; a larger group concatenates each
    update's content blocks in order, so multiple events for one task
    collapse into one turn-context entry instead of one message per event.
    """
    messages = [update.to_message() for update in updates]
    if len(messages) == 1:
        return messages[0]
    content = [block for message in messages for block in message.content]
    return TaskMessage(content=content, created_at=messages[-1].created_at)
