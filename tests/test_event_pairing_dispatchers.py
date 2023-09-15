import uuid
from typing import Any

import pytest

from nebuly.event_pairing_dispatchers import EventPairingDispatcher


def test_new_event_pairing_dispatcher() -> None:
    event_dispatcher = EventPairingDispatcher()
    assert event_dispatcher is not None


@pytest.fixture(name="simple_sequential_chain_serialized_data")
def fixture_simple_sequential_chain_serialized_data() -> dict[str, Any]:
    return {
        "lc": 1,
        "type": "not_implemented",
        "id": ["langchain", "chains", "sequential", "SimpleSequentialChain"],
        "repr": "SimpleSequentialChain(chains=[synopsis_chain, review_chain], "
        "verbose=True)",
    }


@pytest.fixture(name="llm_chain_serialized_data")
def fixture_llm_chain_serialized_data() -> dict[str, Any]:
    return {
        "lc": 1,
        "type": "not_implemented",
        "id": ["langchain", "chains", "llm", "LLMChain"],
        "kwargs": {
            "llm": {
                "lc": 1,
                "type": "constructor",
                "id": ["langchain", "llms", "openai", "OpenAI"],
                "kwargs": {
                    "temperature": 0.7,
                    "openai_api_key": {
                        "lc": 1,
                        "type": "secret",
                        "id": ["OPENAI_API_KEY"],
                    },
                },
            },
            "prompt": {
                "lc": 1,
                "type": "constructor",
                "id": ["langchain", "prompts", "prompt", "PromptTemplate"],
                "kwargs": {
                    "input_variables": ["title"],
                    "template": "You are a playwright. Given the title of play, it is "
                    "your job to write a synopsis for that title.\n\nTitle:"
                    " {title}\nPlaywright: This is a synopsis for the above"
                    " play:",
                    "template_format": "f-string",
                },
            },
        },
    }


def test_event_pairing_dispatcher_on_chain_start__new_root_chain(
    simple_sequential_chain_serialized_data: dict[str, Any]
) -> None:
    run_id = uuid.uuid4()
    event_dispatcher = EventPairingDispatcher()
    event_dispatcher.on_chain_start(
        serialized=simple_sequential_chain_serialized_data,
        inputs={"input": "Tragedy at sunset on the beach"},
        run_id=run_id,
        parent_run_id=None,
    )

    assert event_dispatcher.events_storage.events is not None
    assert len(event_dispatcher.events_storage.events) == 1
    assert run_id in event_dispatcher.events_storage.events
    assert event_dispatcher.events_storage.events[run_id].event_id == run_id
    assert event_dispatcher.events_storage.events[run_id].hierarchy is None
    assert event_dispatcher.events_storage.events[run_id].data == {
        "chain": "SimpleSequentialChain",
        "inputs": {"input": "Tragedy at sunset on the beach"},
    }


def test_event_pairing_dispatcher_on_chain_start__existing_root_chain(
    simple_sequential_chain_serialized_data: dict[str, Any],
    llm_chain_serialized_data: dict[str, Any],
) -> None:
    root_id = uuid.uuid4()
    event_dispatcher = EventPairingDispatcher()
    event_dispatcher.on_chain_start(
        serialized=simple_sequential_chain_serialized_data,
        inputs={"input": "Tragedy at sunset on the beach"},
        run_id=root_id,
        parent_run_id=None,
    )

    run_id = uuid.uuid4()
    event_dispatcher.on_chain_start(
        serialized=llm_chain_serialized_data,
        inputs={"title": "Tragedy at sunset on the beach"},
        run_id=run_id,
        parent_run_id=root_id,
    )

    assert event_dispatcher.events_storage.events is not None
    assert len(event_dispatcher.events_storage.events) == 2
    assert run_id in event_dispatcher.events_storage.events
    assert event_dispatcher.events_storage.events[run_id].event_id == run_id
    assert event_dispatcher.events_storage.events[run_id].hierarchy is not None
    assert event_dispatcher.events_storage.events[run_id].hierarchy.parent_id == root_id  # type: ignore # pylint: disable=line-too-long
    assert event_dispatcher.events_storage.events[run_id].hierarchy.root_id == root_id  # type: ignore # pylint: disable=line-too-long
    assert event_dispatcher.events_storage.events[run_id].data == {
        "chain": "LLMChain",
        "inputs": {"title": "Tragedy at sunset on the beach"},
    }


def test_event_pairing_dispatcher_on_chain_end__root_chain(
    simple_sequential_chain_serialized_data: dict[str, Any]
) -> None:
    run_id = uuid.uuid4()
    event_dispatcher = EventPairingDispatcher()
    event_dispatcher.on_chain_start(
        serialized=simple_sequential_chain_serialized_data,
        inputs={"input": "Tragedy at sunset on the beach"},
        run_id=run_id,
        parent_run_id=None,
    )

    watched_event = event_dispatcher.on_chain_end(
        outputs={"text": "Tragedy at Sunset on the Beach"},
        run_id=run_id,
        parent_run_id=None,
    )

    assert event_dispatcher.events_storage.events is not None
    assert len(event_dispatcher.events_storage.events) == 0

    assert watched_event is not None
    assert watched_event.run_id == run_id
    assert watched_event.parent_run_id is None
    assert watched_event.root_run_id is None
    assert watched_event.inputs == {"input": "Tragedy at sunset on the beach"}
    assert watched_event.outputs == {"text": "Tragedy at Sunset on the Beach"}
    assert watched_event.chain == "SimpleSequentialChain"


def test_event_pairing_dispatcher_on_chain_end__child_chain(
    simple_sequential_chain_serialized_data: dict[str, Any],
    llm_chain_serialized_data: dict[str, Any],
) -> None:
    root_id = uuid.uuid4()
    event_dispatcher = EventPairingDispatcher()
    event_dispatcher.on_chain_start(
        serialized=simple_sequential_chain_serialized_data,
        inputs={"input": "Tragedy at sunset on the beach"},
        run_id=root_id,
        parent_run_id=None,
    )

    run_id = uuid.uuid4()
    event_dispatcher.on_chain_start(
        serialized=llm_chain_serialized_data,
        inputs={"title": "Tragedy at sunset on the beach"},
        run_id=run_id,
        parent_run_id=root_id,
    )

    watched_event = event_dispatcher.on_chain_end(
        outputs={"text": "Tragedy at Sunset on the Beach"},
        run_id=run_id,
        parent_run_id=root_id,
    )

    assert event_dispatcher.events_storage.events is not None
    assert len(event_dispatcher.events_storage.events) == 2
    assert root_id in event_dispatcher.events_storage.events
    assert run_id in event_dispatcher.events_storage.events

    assert watched_event is not None
    assert watched_event.run_id == run_id
    assert watched_event.parent_run_id == root_id
    assert watched_event.root_run_id == root_id
    assert watched_event.inputs == {"title": "Tragedy at sunset on the beach"}
    assert watched_event.outputs == {"text": "Tragedy at Sunset on the Beach"}
    assert watched_event.chain == "LLMChain"
