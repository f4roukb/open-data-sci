"""Unit tests for opendatasci._tui.message_queue.PendingMessageQueue."""

import pytest

from opendatasci._tui.message_queue import PendingMessage, PendingMessageQueue


# ---------------------------------------------------------------------------
# Enqueue
# ---------------------------------------------------------------------------


class TestPendingMessageQueueEnqueue:
    def test_enqueue_creates_message_with_agent_query(self) -> None:
        q = PendingMessageQueue()
        msg = q.enqueue("SELECT * FROM data", "Tell me about the data")
        assert msg.agent_query == "SELECT * FROM data"

    def test_enqueue_creates_message_with_display_text(self) -> None:
        q = PendingMessageQueue()
        msg = q.enqueue("query text", "display text")
        assert msg.display == "display text"

    def test_enqueue_returns_pending_message_instance(self) -> None:
        q = PendingMessageQueue()
        msg = q.enqueue("q", "d")
        assert isinstance(msg, PendingMessage)

    def test_enqueue_assigns_positive_integer_id(self) -> None:
        q = PendingMessageQueue()
        msg = q.enqueue("q", "d")
        assert isinstance(msg.id, int)
        assert msg.id > 0

    def test_enqueue_assigns_strictly_increasing_ids(self) -> None:
        q = PendingMessageQueue()
        m1 = q.enqueue("q1", "d1")
        m2 = q.enqueue("q2", "d2")
        m3 = q.enqueue("q3", "d3")
        assert m2.id > m1.id
        assert m3.id > m2.id

    def test_enqueue_increases_queue_length(self) -> None:
        q = PendingMessageQueue()
        q.enqueue("q1", "d1")
        assert len(q) == 1
        q.enqueue("q2", "d2")
        assert len(q) == 2

    def test_enqueue_marks_queue_non_empty(self) -> None:
        q = PendingMessageQueue()
        q.enqueue("q", "d")
        assert not q.is_empty()

    def test_ids_unique_across_enqueues(self) -> None:
        q = PendingMessageQueue()
        messages = [q.enqueue(f"q{i}", f"d{i}") for i in range(10)]
        ids = [m.id for m in messages]
        assert len(set(ids)) == 10  # all unique


# ---------------------------------------------------------------------------
# pop_next — FIFO order
# ---------------------------------------------------------------------------


class TestPendingMessageQueuePopNext:
    def test_pop_next_on_empty_queue_returns_none(self) -> None:
        q = PendingMessageQueue()
        assert q.pop_next() is None

    def test_pop_next_returns_oldest_message(self) -> None:
        q = PendingMessageQueue()
        m1 = q.enqueue("q1", "d1")
        q.enqueue("q2", "d2")
        result = q.pop_next()
        assert result is m1

    def test_pop_next_removes_message_from_queue(self) -> None:
        q = PendingMessageQueue()
        q.enqueue("q", "d")
        q.pop_next()
        assert q.is_empty()

    def test_pop_next_decreases_length_by_one(self) -> None:
        q = PendingMessageQueue()
        q.enqueue("q1", "d1")
        q.enqueue("q2", "d2")
        q.pop_next()
        assert len(q) == 1

    def test_pop_next_fifo_order_across_multiple_pops(self) -> None:
        q = PendingMessageQueue()
        for i in range(5):
            q.enqueue(f"q{i}", f"d{i}")
        for i in range(5):
            msg = q.pop_next()
            assert msg is not None
            assert msg.agent_query == f"q{i}"

    def test_pop_next_returns_none_after_all_messages_consumed(self) -> None:
        q = PendingMessageQueue()
        q.enqueue("q", "d")
        q.pop_next()
        assert q.pop_next() is None

    def test_pop_next_second_message_becomes_first_after_first_popped(self) -> None:
        q = PendingMessageQueue()
        m1 = q.enqueue("q1", "d1")
        m2 = q.enqueue("q2", "d2")
        q.pop_next()  # removes m1
        result = q.pop_next()
        assert result is m2


# ---------------------------------------------------------------------------
# cancel_all — clear the entire queue
# ---------------------------------------------------------------------------


class TestPendingMessageQueueCancelAll:
    def test_cancel_all_on_empty_returns_empty_list(self) -> None:
        q = PendingMessageQueue()
        assert q.cancel_all() == []

    def test_cancel_all_returns_all_enqueued_messages(self) -> None:
        q = PendingMessageQueue()
        m1 = q.enqueue("q1", "d1")
        m2 = q.enqueue("q2", "d2")
        removed = q.cancel_all()
        assert m1 in removed
        assert m2 in removed

    def test_cancel_all_returns_messages_in_fifo_order(self) -> None:
        q = PendingMessageQueue()
        messages = [q.enqueue(f"q{i}", f"d{i}") for i in range(4)]
        removed = q.cancel_all()
        assert removed == messages

    def test_cancel_all_empties_the_queue(self) -> None:
        q = PendingMessageQueue()
        q.enqueue("q", "d")
        q.cancel_all()
        assert q.is_empty()

    def test_cancel_all_queue_length_is_zero_after(self) -> None:
        q = PendingMessageQueue()
        q.enqueue("q1", "d1")
        q.enqueue("q2", "d2")
        q.cancel_all()
        assert len(q) == 0

    def test_cancel_all_called_twice_second_returns_empty(self) -> None:
        q = PendingMessageQueue()
        q.enqueue("q", "d")
        q.cancel_all()
        assert q.cancel_all() == []


# ---------------------------------------------------------------------------
# cancel_last — remove most-recently-added message
# ---------------------------------------------------------------------------


class TestPendingMessageQueueCancelLast:
    def test_cancel_last_on_empty_returns_none(self) -> None:
        q = PendingMessageQueue()
        assert q.cancel_last() is None

    def test_cancel_last_returns_most_recently_enqueued(self) -> None:
        q = PendingMessageQueue()
        q.enqueue("q1", "d1")
        m2 = q.enqueue("q2", "d2")
        assert q.cancel_last() is m2

    def test_cancel_last_leaves_earlier_messages_intact(self) -> None:
        q = PendingMessageQueue()
        m1 = q.enqueue("q1", "d1")
        q.enqueue("q2", "d2")
        q.cancel_last()
        assert len(q) == 1
        assert q.pop_next() is m1

    def test_cancel_last_called_twice_removes_in_reverse_order(self) -> None:
        q = PendingMessageQueue()
        m1 = q.enqueue("q1", "d1")
        m2 = q.enqueue("q2", "d2")
        assert q.cancel_last() is m2
        assert q.cancel_last() is m1
        assert q.is_empty()

    def test_cancel_last_after_queue_empties_returns_none(self) -> None:
        q = PendingMessageQueue()
        q.enqueue("q", "d")
        q.cancel_last()
        assert q.cancel_last() is None

    def test_cancel_last_decreases_length_by_one(self) -> None:
        q = PendingMessageQueue()
        q.enqueue("q1", "d1")
        q.enqueue("q2", "d2")
        q.cancel_last()
        assert len(q) == 1


# ---------------------------------------------------------------------------
# Queue state — is_empty and __len__
# ---------------------------------------------------------------------------


class TestPendingMessageQueueState:
    def test_is_empty_true_on_fresh_queue(self) -> None:
        q = PendingMessageQueue()
        assert q.is_empty()

    def test_len_zero_on_fresh_queue(self) -> None:
        q = PendingMessageQueue()
        assert len(q) == 0

    def test_is_empty_false_after_enqueue(self) -> None:
        q = PendingMessageQueue()
        q.enqueue("q", "d")
        assert not q.is_empty()

    def test_len_matches_enqueue_count(self) -> None:
        q = PendingMessageQueue()
        for i in range(7):
            q.enqueue(f"q{i}", f"d{i}")
        assert len(q) == 7

    def test_is_empty_true_after_all_popped(self) -> None:
        q = PendingMessageQueue()
        q.enqueue("q1", "d1")
        q.enqueue("q2", "d2")
        q.pop_next()
        q.pop_next()
        assert q.is_empty()

    def test_len_decrements_correctly_on_pop(self) -> None:
        q = PendingMessageQueue()
        q.enqueue("q1", "d1")
        q.enqueue("q2", "d2")
        q.pop_next()
        assert len(q) == 1

    def test_len_after_cancel_last(self) -> None:
        q = PendingMessageQueue()
        q.enqueue("q1", "d1")
        q.enqueue("q2", "d2")
        q.cancel_last()
        assert len(q) == 1

    def test_queue_can_be_refilled_after_cancel_all(self) -> None:
        q = PendingMessageQueue()
        q.enqueue("q1", "d1")
        q.cancel_all()
        q.enqueue("q2", "d2")
        assert len(q) == 1
        assert not q.is_empty()
