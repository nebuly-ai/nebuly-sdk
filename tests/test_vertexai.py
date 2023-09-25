from unittest.mock import patch

import pytest
from google.cloud import aiplatform
from vertexai.language_models import TextGenerationModel, TextGenerationResponse

import nebuly
from nebuly.contextmanager import new_interaction
from nebuly.entities import InteractionWatch, SpanWatch
from nebuly.observers import NebulyObserver


@pytest.fixture()
def palm_completion() -> TextGenerationResponse:
    return TextGenerationResponse(
        is_blocked=False,
        _prediction_response=aiplatform.models.Prediction(
            predictions=[
                {
                    "content": "1. What is your experience with project management?\n2. What are your strengths and weaknesses as a project manager?\n3. How do you handle conflict and difficult situations?\n4. How do you communicate with stakeholders and keep them informed?\n5. How do you manage your time and resources effectively?\n6. What is your approach to risk management?\n7. How do you measure the success of your projects?\n8. What are your goals for the future?\n9. What are your salary expectations?\n10. Why do you want to work for our company?",
                    "citationMetadata": {"citations": []},
                    "safetyAttributes": {
                        "blocked": False,
                        "scores": [0.7, 0.1, 0.1],
                        "categories": ["Finance", "Health", "Toxic"],
                    },
                }
            ],
            deployed_model_id="",
            model_version_id="",
            model_resource_name="",
            explanations=None,
        ),
        safety_attributes={"Finance": 0.7, "Health": 0.1, "Toxic": 0.1},
        text="1. What is your experience with project management?\n2. What are your strengths and weaknesses as a project manager?\n3. How do you handle conflict and difficult situations?\n4. How do you communicate with stakeholders and keep them informed?\n5. How do you manage your time and resources effectively?\n6. What is your approach to risk management?\n7. How do you measure the success of your projects?\n8. What are your goals for the future?\n9. What are your salary expectations?\n10. Why do you want to work for our company?",
    )


def test_vertexai_completion__no_context_manager(palm_completion):
    with patch(
        "vertexai.language_models.TextGenerationModel.predict"
    ) as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = palm_completion
            nebuly.init(api_key="test", disable_checks=True)

            parameters = {
                "temperature": 0,
                # Temperature controls the degree of randomness in token selection.
                "max_output_tokens": 256,
                # Token limit determines the maximum amount of text output.
                "top_p": 0.8,
                # Tokens are selected from most probable to least until the sum of their probabilities equals the top_p value.
                "top_k": 40,
                # A top_k of 1 means the selected token is the most probable among all tokens.
            }

            model = TextGenerationModel.from_pretrained("text-bison@001")
            result = model.predict(
                "Give me ten interview questions for the role of program manager.",
                **parameters,
            )
            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert (
                interaction_watch.input
                == "Give me ten interview questions for the role of program manager."
            )
            assert interaction_watch.output == palm_completion.text
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)


def test_vertexai_completion__with_context_manager(palm_completion):
    with patch(
        "vertexai.language_models.TextGenerationModel.predict"
    ) as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = palm_completion
            nebuly.init(api_key="test", disable_checks=True)

            with new_interaction(
                "Give me ten interview questions for the role of program manager."
            ) as interaction_watch:
                interaction_watch.set_input(
                    "Give me ten interview questions for the role of program manager."
                )
                parameters = {
                    "temperature": 0,
                    # Temperature controls the degree of randomness in token selection.
                    "max_output_tokens": 256,
                    # Token limit determines the maximum amount of text output.
                    "top_p": 0.8,
                    # Tokens are selected from most probable to least until the sum of their probabilities equals the top_p value.
                    "top_k": 40,
                    # A top_k of 1 means the selected token is the most probable among all tokens.
                }
                model = TextGenerationModel.from_pretrained("text-bison@001")
                result = model.predict(
                    "Give me ten interview questions for the role of program manager.",
                    **parameters,
                )
                interaction_watch.set_output(result.text)
            assert result is not None
            assert mock_observer.call_count == 1
            assert interaction_watch.output == palm_completion.text
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
