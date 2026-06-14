"""Component tests: OpenDataSci.astream() happy path."""


import asyncio

from opendatasci.streaming import BaseAgentStreamEvent


class TestStreamQuery:
    """Happy path: streaming a query after loading a file."""

    async def test_astream_yields_token_and_response_events(self, loaded_opendatasci_service):
        events = [e async for e in loaded_opendatasci_service.astream("Describe the data")]
        await asyncio.sleep(0)  # drain background summarisation task

        types = {e.type for e in events}
        assert "token" in types
        assert "response" in types

    async def test_response_event_has_content(self, loaded_opendatasci_service):
        events = [e async for e in loaded_opendatasci_service.astream("What are the trends?")]
        await asyncio.sleep(0)

        response_event = next(e for e in events if e.type == "response")
        assert response_event.content  # non-empty explanation

    async def test_all_yielded_objects_are_stream_events(self, loaded_opendatasci_service):
        events = [e async for e in loaded_opendatasci_service.astream("Summarise")]
        await asyncio.sleep(0)

        assert all(isinstance(e, BaseAgentStreamEvent) for e in events)
