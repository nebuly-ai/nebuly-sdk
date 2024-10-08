# pylint: disable=duplicate-code
from __future__ import annotations

import json
from typing import AsyncGenerator
from unittest.mock import AsyncMock, patch

import pytest
from anthropic import AI_PROMPT, HUMAN_PROMPT, Anthropic, AsyncAnthropic
from anthropic.types import Completion

from nebuly.contextmanager import new_interaction
from nebuly.entities import HistoryEntry, InteractionWatch, SpanWatch
from nebuly.observers import NebulyObserver
from nebuly.requests import CustomJSONEncoder
from tests.providers.common import nebuly_init


@pytest.fixture(name="anthropic_completion")
def fixture_anthropic_completion() -> Completion:
    return Completion(
        id="123",
        completion=" Hello! My name is Claude.",
        stop_reason="stop_sequence",
        model="claude-2",
        type="completion",
    )


def test_anthropic_completion__no_context_manager(
    anthropic_completion: Completion,
) -> None:
    with patch("anthropic.resources.Completions.create") as mock_completion:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion.return_value = anthropic_completion
            nebuly_init(observer=mock_observer)

            client = Anthropic(
                # defaults to os.environ.get("ANTHROPIC_API_KEY")
                api_key="my api key",
            )

            result = client.completions.create(  # type: ignore
                model="claude-2",
                max_tokens_to_sample=300,
                prompt=f"{HUMAN_PROMPT} say hi{AI_PROMPT} hi {HUMAN_PROMPT} "
                f"say hello{AI_PROMPT}",
                user_id="test_user",
                user_group_profile="test_group",
            )
            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "say hello"
            assert interaction_watch.history == [
                HistoryEntry(
                    user="say hi",
                    assistant="hi",
                )
            ]
            assert interaction_watch.output == "Hello! My name is Claude."
            assert interaction_watch.end_user == "test_user"
            assert interaction_watch.end_user_group_profile == "test_group"
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


def test_anthropic_completion__with_context_manager(
    anthropic_completion: Completion,
) -> None:
    with patch("anthropic.resources.Completions.create") as mock_completion:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion.return_value = anthropic_completion
            nebuly_init(observer=mock_observer)

            client = Anthropic(
                # defaults to os.environ.get("ANTHROPIC_API_KEY")
                api_key="my api key",
            )

            with new_interaction(
                user_id="test_user", user_group_profile="test_group"
            ) as interaction:
                interaction.set_input("Sample input 1")
                result = client.completions.create(
                    model="claude-2",
                    max_tokens_to_sample=300,
                    prompt=f"{HUMAN_PROMPT} how does a court case get to the "
                    f"Supreme Court?{AI_PROMPT}",
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
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


@pytest.mark.asyncio
async def test_anthropic_completion__async(anthropic_completion: Completion) -> None:
    with patch(
        "anthropic.resources.AsyncCompletions.create", new=AsyncMock()
    ) as mock_completion:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion.return_value = anthropic_completion
            nebuly_init(observer=mock_observer)

            client = AsyncAnthropic(
                # defaults to os.environ.get("ANTHROPIC_API_KEY")
                api_key="my api key",
            )

            result = await client.completions.create(  # type: ignore
                model="claude-2",
                max_tokens_to_sample=300,
                prompt=f"{HUMAN_PROMPT} how does a court case get to the "
                f"Supreme Court?{AI_PROMPT}",
                user_id="test_user",
                user_group_profile="test_group",
            )
            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert (
                interaction_watch.input
                == "how does a court case get to the Supreme Court?"
            )
            assert interaction_watch.output == "Hello! My name is Claude."
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


@pytest.fixture(name="anthropic_completion_gen")
def fixture_anthropic_completion_gen() -> list[Completion]:
    return [
        Completion(
            id="123",
            completion=" Hello",
            stop_reason="stop_sequence",
            model="claude-2",
            type="completion",
        ),
        Completion(
            id="123",
            completion="!",
            stop_reason="stop_sequence",
            model="claude-2",
            type="completion",
        ),
    ]


def test_anthropic_completion_gen(anthropic_completion_gen: list[Completion]) -> None:
    with patch("anthropic.resources.Completions.create") as mock_completion:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion.return_value = (
                completion for completion in anthropic_completion_gen
            )
            nebuly_init(observer=mock_observer)

            client = Anthropic(
                # defaults to os.environ.get("ANTHROPIC_API_KEY")
                api_key="my api key",
            )

            for _ in client.completions.create(  # type: ignore  # pylint: disable=not-an-iterable  # noqa: E501
                model="claude-2",
                max_tokens_to_sample=300,
                prompt=f"{HUMAN_PROMPT} how does a court case get to the "
                f"Supreme Court?{AI_PROMPT}",
                stream=True,
                user_id="test_user",
                user_group_profile="test_group",
            ):
                ...
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert (
                interaction_watch.input
                == "how does a court case get to the Supreme Court?"
            )
            assert interaction_watch.output == "Hello!"
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


@pytest.fixture(name="anthropic_completion_gen_async")
async def fixture_anthropic_completion_gen_async() -> AsyncGenerator[Completion, None]:
    chunks = [
        Completion(
            id="123",
            completion=" Hello",
            stop_reason="stop_sequence",
            model="claude-2",
            type="completion",
        ),
        Completion(
            id="123",
            completion="!",
            stop_reason="stop_sequence",
            model="claude-2",
            type="completion",
        ),
    ]

    for chunk in chunks:
        yield chunk


@pytest.mark.asyncio
async def test_anthropic_completion_async_gen(
    anthropic_completion_gen_async: AsyncGenerator[Completion, None]
) -> None:
    with patch(
        "anthropic.resources.AsyncCompletions.create", new=AsyncMock()
    ) as mock_completion:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion.return_value = anthropic_completion_gen_async
            nebuly_init(observer=mock_observer)

            client = AsyncAnthropic(
                # defaults to os.environ.get("ANTHROPIC_API_KEY")
                api_key="my api key",
            )

            async for _ in await client.completions.create(  # type: ignore  # pylint: disable=not-an-iterable  # noqa: E501
                model="claude-2",
                max_tokens_to_sample=300,
                prompt=f"{HUMAN_PROMPT} how does a court case get to the "
                f"Supreme Court?{AI_PROMPT}",
                stream=True,
                user_id="test_user",
                user_group_profile="test_group",
            ):
                ...
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert (
                interaction_watch.input
                == "how does a court case get to the Supreme Court?"
            )
            assert interaction_watch.output == "Hello!"
            assert interaction_watch.end_user == "test_user"
            assert interaction_watch.end_user_group_profile == "test_group"
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )
