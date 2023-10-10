from __future__ import annotations

import json
from unittest.mock import patch
from uuid import uuid4

import pytest
from transformers import Conversation, pipeline  # type: ignore

from nebuly.entities import InteractionWatch, SpanWatch
from nebuly.observers import NebulyObserver
from nebuly.requests import CustomJSONEncoder
from tests.common import nebuly_init


@pytest.fixture(name="hf_conversational_pipelines")
def fixture_hf_conversational_pipelines() -> Conversation:
    conversation = Conversation(
        uuid=uuid4(),
    )
    conversation.add_user_input("Going to the movies tonight - any suggestions?")
    conversation.add_message({"content": "The Big Lebowski", "role": "assistant"})

    return conversation


def test_hf_conversational_pipelines__single_prompt__with_history(
    hf_conversational_pipelines: Conversation,
) -> None:
    with patch("transformers.pipelines.base.Pipeline.__call__") as mock_conversational:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_conversational.return_value = hf_conversational_pipelines
            nebuly_init(observer=mock_observer)

            converse = pipeline(task="conversational")
            conversation = Conversation(
                [
                    {
                        "content": "Going to the movies tonight - any suggestions?",
                        "role": "user",
                    },
                    {"content": "The Big Lebowski", "role": "assistant"},
                    {
                        "content": "Going to the movies also tomorrow - any "
                        "suggestions?",
                        "role": "user",
                    },
                ]
            )
            result = converse(
                conversation, user_id="test_user", user_group_profile="test_group"
            )

            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert (
                interaction_watch.input
                == "Going to the movies also tomorrow - any suggestions?"
            )
            assert interaction_watch.history == [
                (
                    "Going to the movies tonight - any suggestions?",
                    "The Big Lebowski",
                )
            ]
            assert interaction_watch.output == "The Big Lebowski"
            assert interaction_watch.end_user == "test_user"
            assert interaction_watch.end_user_group_profile == "test_group"
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


@pytest.fixture(name="hf_conversational_pipelines_batch")
def fixture_hf_conversational_pipelines_batch() -> list[Conversation]:
    conversation = Conversation(
        uuid=uuid4(),
    )
    conversation.add_user_input("Going to the movies tonight - any suggestions?")
    conversation.add_message({"content": "The Big Lebowski", "role": "assistant"})

    return [conversation, conversation]


def test_hf_conversational_pipelines__prompt_list(
    hf_conversational_pipelines_batch: list[Conversation],
) -> None:
    with patch("transformers.pipelines.base.Pipeline.__call__") as mock_conversational:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_conversational.return_value = hf_conversational_pipelines_batch
            nebuly_init(observer=mock_observer)

            converse = pipeline(task="conversational")
            conversation_1 = Conversation(
                "Going to the movies tonight - any suggestions?"
            )
            conversation_2 = Conversation("What's the last book you have read?")
            result = converse(
                [conversation_1, conversation_2],
                user_id="test_user",
                user_group_profile="test_group",
            )

            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert (
                interaction_watch.input
                == "Going to the movies tonight - any suggestions?"
            )
            assert interaction_watch.history == []
            assert interaction_watch.output == "The Big Lebowski"
            assert interaction_watch.end_user == "test_user"
            assert interaction_watch.end_user_group_profile == "test_group"
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


@pytest.fixture(name="hf_text_generation_pipelines")
def fixture_hf_text_generation_pipelines() -> list[dict[str, str]]:
    return [{"generated_text": "The Big Lebowski"}]


def test_hf_text_generation_pipelines__single_prompt(
    hf_text_generation_pipelines: list[dict[str, str]]
) -> None:
    with patch("transformers.pipelines.base.Pipeline.__call__") as mock_text_generation:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_text_generation.return_value = hf_text_generation_pipelines
            nebuly_init(observer=mock_observer)

            user_input = "As far as I am concerned, I will"
            generator = pipeline("text-generation")
            result = generator(
                user_input,
                user_id="test_user",
                user_group_profile="test_group",
            )

            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == user_input
            assert interaction_watch.history == []
            assert interaction_watch.output == "The Big Lebowski"
            assert interaction_watch.end_user == "test_user"
            assert interaction_watch.end_user_group_profile == "test_group"
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


def test_hf_text_generation_pipelines__single_prompt__multiple_return_sequences(
    hf_text_generation_pipelines: list[dict[str, str]]
) -> None:
    with patch("transformers.pipelines.base.Pipeline.__call__") as mock_text_generation:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_text_generation.return_value = [
                hf_text_generation_pipelines[0],
                hf_text_generation_pipelines[0],
            ]
            nebuly_init(observer=mock_observer)

            user_input = "As far as I am concerned, I will"
            generator = pipeline("text-generation")
            result = generator(
                user_input,
                num_return_sequences=2,
                user_id="test_user",
                user_group_profile="test_group",
            )

            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == user_input
            assert interaction_watch.history == []
            assert interaction_watch.output == "The Big Lebowski"
            assert interaction_watch.end_user == "test_user"
            assert interaction_watch.end_user_group_profile == "test_group"
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


@pytest.fixture(name="hf_text_generation_pipelines_batch")
def fixture_hf_text_generation_pipelines_batch() -> list[list[dict[str, str]]]:
    return [
        [{"generated_text": "The Big Lebowski"}],
        [{"generated_text": "The Big Lebowski"}],
    ]


def test_hf_text_generation_pipelines__batch_prompt(
    hf_text_generation_pipelines_batch: list[list[dict[str, str]]]
) -> None:
    with patch("transformers.pipelines.base.Pipeline.__call__") as mock_text_generation:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_text_generation.return_value = hf_text_generation_pipelines_batch
            nebuly_init(observer=mock_observer)

            user_input = "As far as I am concerned, I will"
            generator = pipeline("text-generation")
            result = generator(
                [user_input, user_input],
                user_id="test_user",
                user_group_profile="test_group",
            )

            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == user_input
            assert interaction_watch.history == []
            assert interaction_watch.output == "The Big Lebowski"
            assert interaction_watch.end_user == "test_user"
            assert interaction_watch.end_user_group_profile == "test_group"
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


def test_nebuly_init_does_not_break_other_pipelines() -> None:
    with patch.object(NebulyObserver, "on_event_received") as mock_observer:
        nebuly_init(observer=mock_observer)

        generator = pipeline("text-classification")
        result = generator("As far as I am concerned, I will")

        assert result is not None
