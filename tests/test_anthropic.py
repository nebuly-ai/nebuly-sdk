from unittest.mock import AsyncMock, patch

import pytest
from anthropic import AI_PROMPT, HUMAN_PROMPT, Anthropic, AsyncAnthropic
from anthropic.types import Completion

import nebuly
from nebuly.contextmanager import new_interaction
from nebuly.entities import InteractionWatch, SpanWatch
from nebuly.observers import NebulyObserver


@pytest.fixture()
def anthropic_completion() -> Completion:
    return Completion(
        completion=" Hello! My name is Claude.",
        stop_reason="stop_sequence",
        model="claude-2",
    )


def test_anthropic_completion__no_context_manager(anthropic_completion):
    with patch("anthropic.resources.Completions.create") as mock_completion:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion.return_value = anthropic_completion
            nebuly.init(api_key="test")

            client = Anthropic(
                # defaults to os.environ.get("ANTHROPIC_API_KEY")
                api_key="my api key",
            )

            result = client.completions.create(
                model="claude-2",
                max_tokens_to_sample=300,
                prompt=f"{HUMAN_PROMPT} how does a court case get to the Supreme Court?{AI_PROMPT}",
            )
            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert (
                interaction_watch.input
                == f"{HUMAN_PROMPT} how does a court case get to the Supreme Court?{AI_PROMPT}"
            )
            assert interaction_watch.output == " Hello! My name is Claude."
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)


def test_anthropic_completion__with_context_manager(anthropic_completion):
    with patch("anthropic.resources.Completions.create") as mock_completion:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion.return_value = anthropic_completion
            nebuly.init(api_key="test")

            client = Anthropic(
                # defaults to os.environ.get("ANTHROPIC_API_KEY")
                api_key="my api key",
            )

            with new_interaction(
                user="test_user", group_profile="test_group"
            ) as interaction:
                interaction.set_input("Sample input 1")
                result = client.completions.create(
                    model="claude-2",
                    max_tokens_to_sample=300,
                    prompt=f"{HUMAN_PROMPT} how does a court case get to the Supreme Court?{AI_PROMPT}",
                )
                interaction.set_output("Sample output 1")
            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "Sample input 1"
            assert interaction_watch.output == "Sample output 1"
            assert interaction_watch.end_user == "test_user"
            assert interaction_watch.end_user_group_profile == "test_group"
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)


@pytest.mark.asyncio
async def test_anthropic_completion__async(anthropic_completion):
    with patch(
        "anthropic.resources.AsyncCompletions.create", new=AsyncMock()
    ) as mock_completion:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion.return_value = anthropic_completion
            nebuly.init(api_key="test")

            client = AsyncAnthropic(
                # defaults to os.environ.get("ANTHROPIC_API_KEY")
                api_key="my api key",
            )

            result = await client.completions.create(
                model="claude-2",
                max_tokens_to_sample=300,
                prompt=f"{HUMAN_PROMPT} how does a court case get to the Supreme Court?{AI_PROMPT}",
            )
            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert (
                interaction_watch.input
                == f"{HUMAN_PROMPT} how does a court case get to the Supreme Court?{AI_PROMPT}"
            )
            assert interaction_watch.output == " Hello! My name is Claude."
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)


@pytest.fixture()
def anthropic_completion_gen():
    return [
        Completion(
            completion=" Hello",
            stop_reason="stop_sequence",
            model="claude-2",
        ),
        Completion(
            completion="!",
            stop_reason="stop_sequence",
            model="claude-2",
        ),
    ]


def test_anthropic_completion_gen(anthropic_completion_gen):
    with patch("anthropic.resources.Completions.create") as mock_completion:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion.return_value = (
                completion for completion in anthropic_completion_gen
            )
            nebuly.init(api_key="test")

            client = Anthropic(
                # defaults to os.environ.get("ANTHROPIC_API_KEY")
                api_key="my api key",
            )

            for _ in client.completions.create(
                model="claude-2",
                max_tokens_to_sample=300,
                prompt=f"{HUMAN_PROMPT} how does a court case get to the Supreme Court?{AI_PROMPT}",
                stream=True,
            ):
                ...
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert (
                interaction_watch.input
                == f"{HUMAN_PROMPT} how does a court case get to the Supreme Court?{AI_PROMPT}"
            )
            assert interaction_watch.output == " Hello!"
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
