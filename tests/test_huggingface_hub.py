from unittest.mock import patch

import pytest
from huggingface_hub import InferenceClient

import nebuly
from nebuly.contextmanager import new_interaction
from nebuly.entities import InteractionWatch, SpanWatch
from nebuly.observers import NebulyObserver


@pytest.fixture()
def hf_hub_sample_input() -> dict:
    return {
        "generated_text": " My name is samantha, and I am a student "
        "at the University of Pittsburgh.",
        "conversation": {
            "generated_responses": [
                " My name is samantha, and I am a student at "
                "the University of Pittsburgh."
            ],
            "past_user_inputs": ["Hello, who are you?"],
        },
    }


@pytest.fixture()
def hf_hub_conversational() -> dict:
    return {
        "generated_text": " It is, but I am studying to be a nurse. What do you do?",
        "conversation": {
            "generated_responses": [
                " My name is samantha, and I am a student at the "
                "University of Pittsburgh.",
                " It is, but I am studying to be a nurse. What do you do?",
            ],
            "past_user_inputs": ["Hello, who are you?", "Wow, that's scary!"],
        },
    }


def test_hf_hub_conversational__no_context_manager__no_history(
    hf_hub_sample_input, hf_hub_conversational
):
    with patch("huggingface_hub.InferenceClient.conversational") as mock_conversational:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_conversational.return_value = hf_hub_conversational
            nebuly.init(api_key="test", disable_checks=True)

            client = InferenceClient()
            result = client.conversational(
                text="Wow, that's scary!",
            )

            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "Wow, that's scary!"
            assert interaction_watch.history == []
            assert (
                interaction_watch.output
                == " It is, but I am studying to be a nurse. What do you do?"
            )
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)


def test_hf_hub_conversational__no_context_manager__with_history(
    hf_hub_sample_input, hf_hub_conversational
):
    with patch("huggingface_hub.InferenceClient.conversational") as mock_conversational:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_conversational.return_value = hf_hub_conversational
            nebuly.init(api_key="test", disable_checks=True)

            client = InferenceClient()
            result = client.conversational(
                "Wow, that's scary!",
                generated_responses=hf_hub_sample_input["conversation"][
                    "generated_responses"
                ],
                past_user_inputs=hf_hub_sample_input["conversation"][
                    "past_user_inputs"
                ],
            )

            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "Wow, that's scary!"
            assert interaction_watch.history == [
                ("user", "Hello, who are you?"),
                (
                    "assistant",
                    " My name is samantha, and I am a student at "
                    "the University of Pittsburgh.",
                ),
            ]
            assert (
                interaction_watch.output
                == " It is, but I am studying to be a nurse. What do you do?"
            )
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)


def test_hf_hub_conversational__with_context_manager__no_history(
    hf_hub_sample_input, hf_hub_conversational
):
    with patch("huggingface_hub.InferenceClient.conversational") as mock_conversational:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_conversational.return_value = hf_hub_conversational
            nebuly.init(api_key="test", disable_checks=True)
            client = InferenceClient()
            with new_interaction(
                user="test_user", group_profile="test_group"
            ) as interaction:
                interaction.set_input("Wow, that's scary!")
                result = client.conversational(
                    "Wow, that's scary!",
                )
                interaction.set_output(
                    " It is, but I am studying to be a nurse. What do you do?"
                )

            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "Wow, that's scary!"
            assert interaction_watch.history == []
            assert (
                interaction_watch.output
                == " It is, but I am studying to be a nurse. What do you do?"
            )
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)


def test_hf_hub_conversational__with_context_manager__with_history(
    hf_hub_sample_input, hf_hub_conversational
):
    with patch("huggingface_hub.InferenceClient.conversational") as mock_conversational:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_conversational.return_value = hf_hub_conversational
            nebuly.init(api_key="test", disable_checks=True)
            client = InferenceClient()
            with new_interaction(
                user="test_user", group_profile="test_group"
            ) as interaction:
                interaction.set_input("Wow, that's scary!")
                interaction.set_history(
                    [
                        ("user", "Hello, who are you?"),
                        (
                            "assistant",
                            " My name is samantha, and I am a student at "
                            "the University of Pittsburgh.",
                        ),
                    ]
                )
                result = client.conversational(
                    "Wow, that's scary!",
                    generated_responses=hf_hub_sample_input["conversation"][
                        "generated_responses"
                    ],
                    past_user_inputs=hf_hub_sample_input["conversation"][
                        "past_user_inputs"
                    ],
                )
                interaction.set_output(
                    " It is, but I am studying to be a nurse. What do you do?"
                )

            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "Wow, that's scary!"
            assert interaction_watch.history == [
                ("user", "Hello, who are you?"),
                (
                    "assistant",
                    " My name is samantha, and I am a student at "
                    "the University of Pittsburgh.",
                ),
            ]
            assert (
                interaction_watch.output
                == " It is, but I am studying to be a nurse. What do you do?"
            )
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
