from dataclasses import dataclass
from unittest.mock import patch

import google.generativeai as palm
import pytest
from google.generativeai.types.safety_types import (
    ContentFilterDict,
    HarmCategory,
    HarmProbability,
    SafetyFeedbackDict,
)
from google.generativeai.types.text_types import TextCompletion

import nebuly
from nebuly.contextmanager import new_interaction
from nebuly.entities import InteractionWatch, SpanWatch
from nebuly.observers import NebulyObserver


@dataclass()
class Completion:
    candidates: list[TextCompletion]
    result: str | None
    filters: list[ContentFilterDict | None]
    safety_feedback: list[SafetyFeedbackDict | None]


@pytest.fixture()
def palm_completion() -> Completion:
    return Completion(
        candidates=[
            {
                "output": "cold.",
                "safety_ratings": [
                    {
                        "category": HarmCategory.HARM_CATEGORY_DEROGATORY,
                        "probability": HarmProbability.NEGLIGIBLE,
                    },
                    {
                        "category": HarmCategory.HARM_CATEGORY_TOXICITY,
                        "probability": HarmProbability.NEGLIGIBLE,
                    },
                    {
                        "category": HarmCategory.HARM_CATEGORY_VIOLENCE,
                        "probability": HarmProbability.NEGLIGIBLE,
                    },
                    {
                        "category": HarmCategory.HARM_CATEGORY_SEXUAL,
                        "probability": HarmProbability.NEGLIGIBLE,
                    },
                    {
                        "category": HarmCategory.HARM_CATEGORY_MEDICAL,
                        "probability": HarmProbability.NEGLIGIBLE,
                    },
                    {
                        "category": HarmCategory.HARM_CATEGORY_DANGEROUS,
                        "probability": HarmProbability.NEGLIGIBLE,
                    },
                ],
            }
        ],
        result="cold.",
        filters=[],
        safety_feedback=[],
    )


def test_openai_completion__no_context_manager(palm_completion):
    with patch("google.generativeai.generate_text") as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = palm_completion
            nebuly.init(api_key="test", disable_checks=True)
            palm.configure(api_key="AIzaSyDuQlj_CeVZUFOj8oRm_dwotC4ucAElF_w")

            result = palm.generate_text(prompt="The opposite of hot is")
            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "The opposite of hot is"
            assert interaction_watch.output == "cold."
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)


def test_openai_completion__with_context_manager(palm_completion):
    with patch("google.generativeai.generate_text") as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = palm_completion
            nebuly.init(api_key="test", disable_checks=True)
            palm.configure(api_key="AIzaSyDuQlj_CeVZUFOj8oRm_dwotC4ucAElF_w")

            with new_interaction(
                user="test_user", group_profile="test_group"
            ) as interaction:
                interaction.set_input("Another input")
                result = palm.generate_text(prompt="The opposite of hot is")
                interaction.set_output("Another output")

            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch: InteractionWatch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "Another input"
            assert interaction_watch.output == "Another output"
            assert interaction.user == "test_user"
            assert interaction.user_group_profile == "test_group"
            assert len(interaction_watch.spans) == 1
            span: SpanWatch = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
