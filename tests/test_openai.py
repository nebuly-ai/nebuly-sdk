# pylint: disable=duplicate-code, import-error, no-name-in-module, wrong-import-position, wrong-import-order  # noqa: E501
from __future__ import annotations

import json
from importlib.metadata import version
from typing import AsyncGenerator
from unittest.mock import AsyncMock, patch

import pytest

if version("openai").startswith("0."):
    pytest.skip(
        "Tests enabled only when openai version is 1.X", allow_module_level=True
    )

import openai
from openai import AsyncOpenAI, OpenAI  # mypy: ignore-errors
from openai.types import (  # mypy: ignore-errors
    Completion,
    CompletionChoice,
    CompletionUsage,
)
from openai.types.chat import (  # mypy: ignore-errors
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    chat_completion_chunk,
)
from openai.types.chat.chat_completion import Choice  # mypy: ignore-errors

from nebuly.contextmanager import new_interaction
from nebuly.entities import InteractionWatch, SpanWatch
from nebuly.observers import NebulyObserver
from nebuly.requests import CustomJSONEncoder
from tests.common import nebuly_init


@pytest.fixture(name="openai_completion")
def fixture_openai_completion() -> Completion:
    return Completion(
        id="cmpl-86EVWjLg0rentPFmNfvt5ZlVjKsWW",
        choices=[
            CompletionChoice(
                finish_reason="stop",
                index=0,
                logprobs=None,
                text="\n\nThis is a test.",
            )
        ],
        created=1696496426,
        model="gpt-3.5-turbo-instruct",
        object="text_completion",
        usage=CompletionUsage(completion_tokens=7, prompt_tokens=5, total_tokens=12),
    )


def test_openai_completion__old_format(openai_completion: Completion) -> None:
    with patch(
        "openai.resources.completions.Completions.create"
    ) as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = openai_completion
            nebuly_init(observer=mock_observer)

            result = openai.completions.create(  # type: ignore
                model="gpt-3.5-turbo-instruct",
                prompt="Say this is a test",
                user_id="test_user",
                user_group_profile="test_group",
            )
            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "Say this is a test"
            assert interaction_watch.output == "\n\nThis is a test."
            assert interaction_watch.end_user == "test_user"
            assert interaction_watch.end_user_group_profile == "test_group"
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


def test_openai_completion__no_context_manager(openai_completion: Completion) -> None:
    with patch(
        "openai.resources.completions.Completions.create"
    ) as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = openai_completion
            nebuly_init(observer=mock_observer)

            client = OpenAI(
                api_key="ciao",
            )
            result = client.completions.create(  # type: ignore
                model="gpt-3.5-turbo-instruct",
                prompt="Say this is a test",
                user_id="test_user",
                user_group_profile="test_group",
            )
            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "Say this is a test"
            assert interaction_watch.output == "\n\nThis is a test."
            assert interaction_watch.end_user == "test_user"
            assert interaction_watch.end_user_group_profile == "test_group"
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


def test_openai_completion__context_manager(openai_completion: Completion) -> None:
    with patch(
        "openai.resources.completions.Completions.create"
    ) as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = openai_completion
            nebuly_init(observer=mock_observer)

            client = OpenAI(
                api_key="ciao",
            )

            with new_interaction(
                user_id="test_user", user_group_profile="test_group"
            ) as interaction:
                interaction.set_input("Say this is a test")
                result = client.completions.create(
                    model="gpt-3.5-turbo-instruct",
                    prompt="Say this is a test",
                )
            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "Say this is a test"
            assert interaction_watch.output == "\n\nThis is a test."
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
async def test_openai_completion__async(openai_completion: Completion) -> None:
    with patch(
        "openai.resources.completions.AsyncCompletions.create", new=AsyncMock()
    ) as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = openai_completion
            nebuly_init(observer=mock_observer)

            client = AsyncOpenAI(
                api_key="ciao",
            )
            result = await client.completions.create(  # type: ignore
                model="gpt-3.5-turbo-instruct",
                prompt="Say this is a test",
                user_id="test_user",
                user_group_profile="test_group",
            )
            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "Say this is a test"
            assert interaction_watch.output == "\n\nThis is a test."
            assert interaction_watch.end_user == "test_user"
            assert interaction_watch.end_user_group_profile == "test_group"
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


@pytest.fixture(name="openai_completion_stream")
def fixture_openai_completion_stream() -> list[Completion]:
    return [
        Completion(
            id="cmpl-86FiOh8n7Hjnh4FLNcDe9iuXk7eVf",
            choices=[
                CompletionChoice(
                    finish_reason="stop", index=0, logprobs=None, text="\n\n"
                )
            ],
            created=1696501068,
            model="gpt-3.5-turbo-instruct",
            object="text_completion",
            usage=None,
        ),
        Completion(
            id="cmpl-86FiOh8n7Hjnh4FLNcDe9iuXk7eVf",
            choices=[
                CompletionChoice(
                    finish_reason="stop", index=0, logprobs=None, text="This"
                )
            ],
            created=1696501068,
            model="gpt-3.5-turbo-instruct",
            object="text_completion",
            usage=None,
        ),
        Completion(
            id="cmpl-86FiOh8n7Hjnh4FLNcDe9iuXk7eVf",
            choices=[
                CompletionChoice(
                    finish_reason="stop", index=0, logprobs=None, text=" is"
                )
            ],
            created=1696501068,
            model="gpt-3.5-turbo-instruct",
            object="text_completion",
            usage=None,
        ),
        Completion(
            id="cmpl-86FiOh8n7Hjnh4FLNcDe9iuXk7eVf",
            choices=[
                CompletionChoice(
                    finish_reason="stop", index=0, logprobs=None, text=" a"
                )
            ],
            created=1696501068,
            model="gpt-3.5-turbo-instruct",
            object="text_completion",
            usage=None,
        ),
        Completion(
            id="cmpl-86FiOh8n7Hjnh4FLNcDe9iuXk7eVf",
            choices=[
                CompletionChoice(
                    finish_reason="stop", index=0, logprobs=None, text=" test"
                )
            ],
            created=1696501068,
            model="gpt-3.5-turbo-instruct",
            object="text_completion",
            usage=None,
        ),
        Completion(
            id="cmpl-86FiOh8n7Hjnh4FLNcDe9iuXk7eVf",
            choices=[
                CompletionChoice(finish_reason="stop", index=0, logprobs=None, text=".")
            ],
            created=1696501068,
            model="gpt-3.5-turbo-instruct",
            object="text_completion",
            usage=None,
        ),
        Completion(
            id="cmpl-86FiOh8n7Hjnh4FLNcDe9iuXk7eVf",
            choices=[
                CompletionChoice(finish_reason="stop", index=0, logprobs=None, text="")
            ],
            created=1696501068,
            model="gpt-3.5-turbo-instruct",
            object="text_completion",
            usage=None,
        ),
    ]


def test_openai_completion__stream(openai_completion_stream: list[Completion]) -> None:
    with patch(
        "openai.resources.completions.Completions.create"
    ) as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = (
                el for el in openai_completion_stream
            )
            nebuly_init(observer=mock_observer)

            client = OpenAI(
                api_key="ciao",
            )
            for _ in client.completions.create(  # type: ignore  # pylint: disable=not-an-iterable  # noqa: E501
                model="gpt-3.5-turbo-instruct",
                prompt="Say this is a test",
                stream=True,
                user_id="test_user",
                user_group_profile="test_group",
            ):
                ...
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "Say this is a test"
            assert interaction_watch.output == "\n\nThis is a test."
            assert interaction_watch.end_user == "test_user"
            assert interaction_watch.end_user_group_profile == "test_group"
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


@pytest.fixture(name="openai_completion_stream_async")
async def fixture_openai_completion_stream_async(
    openai_completion_stream: list[Completion],
) -> AsyncGenerator[Completion, None]:
    for el in openai_completion_stream:
        yield el


@pytest.mark.asyncio
async def test_openai_completion__stream_async(
    openai_completion_stream_async: AsyncGenerator[Completion, None]
) -> None:
    with patch(
        "openai.resources.completions.AsyncCompletions.create", new=AsyncMock()
    ) as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = openai_completion_stream_async
            nebuly_init(observer=mock_observer)

            client = AsyncOpenAI(
                api_key="ciao",
            )
            async for _ in await client.completions.create(  # type: ignore
                model="gpt-3.5-turbo-instruct",
                prompt="Say this is a test",
                stream=True,
                user_id="test_user",
                user_group_profile="test_group",
            ):
                ...
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "Say this is a test"
            assert interaction_watch.output == "\n\nThis is a test."
            assert interaction_watch.end_user == "test_user"
            assert interaction_watch.end_user_group_profile == "test_group"
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


@pytest.fixture(name="openai_chat_completion")
def fixture_openai_chat_completion() -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-86HFm0Y1fWK3XOKb0vQ7j09ZUdPws",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    content="\n\nThis is a test.", role="assistant", function_call=None
                ),
            )
        ],
        created=1696506982,
        model="gpt-3.5-turbo-0613",
        object="chat.completion",
        usage=CompletionUsage(completion_tokens=5, prompt_tokens=12, total_tokens=17),
    )


def test_openai_chat__old_format(openai_chat_completion: ChatCompletion) -> None:
    with patch(
        "openai.resources.chat.completions.Completions.create"
    ) as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = openai_chat_completion
            nebuly_init(observer=mock_observer)

            result = openai.chat.completions.create(  # type: ignore
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": "Say this is a test",
                    }
                ],
                user_id="test_user",
                user_group_profile="test_group",
            )
            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "Say this is a test"
            assert interaction_watch.output == "\n\nThis is a test."
            assert interaction_watch.end_user == "test_user"
            assert interaction_watch.end_user_group_profile == "test_group"
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


def test_openai_chat__no_context_manager(
    openai_chat_completion: ChatCompletion,
) -> None:
    with patch(
        "openai.resources.chat.completions.Completions.create"
    ) as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = openai_chat_completion
            nebuly_init(observer=mock_observer)

            client = OpenAI(
                api_key="ciao",
            )
            result = client.chat.completions.create(  # type: ignore
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": "Say this is a test",
                    }
                ],
                user_id="test_user",
                user_group_profile="test_group",
            )
            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "Say this is a test"
            assert interaction_watch.output == "\n\nThis is a test."
            assert interaction_watch.end_user == "test_user"
            assert interaction_watch.end_user_group_profile == "test_group"
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


def test_openai_chat__context_manager(openai_chat_completion: ChatCompletion) -> None:
    with patch(
        "openai.resources.chat.completions.Completions.create"
    ) as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = openai_chat_completion
            nebuly_init(observer=mock_observer)

            client = OpenAI(
                api_key="ciao",
            )

            with new_interaction(
                user_id="test_user", user_group_profile="test_group"
            ) as interaction:
                interaction.set_input("Say this is a test")
                result = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "user",
                            "content": "Say this is a test",
                        }
                    ],
                )
            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "Say this is a test"
            assert interaction_watch.output == "\n\nThis is a test."
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
async def test_openai_chat__async(openai_chat_completion: ChatCompletion) -> None:
    with patch(
        "openai.resources.chat.completions.AsyncCompletions.create", new=AsyncMock()
    ) as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = openai_chat_completion
            nebuly_init(observer=mock_observer)

            client = AsyncOpenAI(
                api_key="ciao",
            )
            result = await client.chat.completions.create(  # type: ignore
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": "Say this is a test",
                    }
                ],
                user_id="test_user",
                user_group_profile="test_group",
            )
            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "Say this is a test"
            assert interaction_watch.output == "\n\nThis is a test."
            assert interaction_watch.end_user == "test_user"
            assert interaction_watch.end_user_group_profile == "test_group"
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


@pytest.fixture(name="openai_chat_completion_stream")
def fixture_openai_chat_completion_stream() -> list[ChatCompletionChunk]:
    return [
        ChatCompletionChunk(
            id="chatcmpl-86HPTGAKCDFXJwEM68ekpoBumMh4H",
            choices=[
                chat_completion_chunk.Choice(
                    delta=chat_completion_chunk.ChoiceDelta(
                        content="", function_call=None, role="assistant"
                    ),
                    finish_reason="stop",
                    index=0,
                )
            ],
            created=1696507583,
            model="gpt-3.5-turbo-0613",
            object="chat.completion.chunk",
        ),
        ChatCompletionChunk(
            id="chatcmpl-86HPTGAKCDFXJwEM68ekpoBumMh4H",
            choices=[
                chat_completion_chunk.Choice(
                    delta=chat_completion_chunk.ChoiceDelta(
                        content="This", function_call=None, role=None
                    ),
                    finish_reason="stop",
                    index=0,
                )
            ],
            created=1696507583,
            model="gpt-3.5-turbo-0613",
            object="chat.completion.chunk",
        ),
        ChatCompletionChunk(
            id="chatcmpl-86HPTGAKCDFXJwEM68ekpoBumMh4H",
            choices=[
                chat_completion_chunk.Choice(
                    delta=chat_completion_chunk.ChoiceDelta(
                        content=" is", function_call=None, role=None
                    ),
                    finish_reason="stop",
                    index=0,
                )
            ],
            created=1696507583,
            model="gpt-3.5-turbo-0613",
            object="chat.completion.chunk",
        ),
        ChatCompletionChunk(
            id="chatcmpl-86HPTGAKCDFXJwEM68ekpoBumMh4H",
            choices=[
                chat_completion_chunk.Choice(
                    delta=chat_completion_chunk.ChoiceDelta(
                        content=" a", function_call=None, role=None
                    ),
                    finish_reason="stop",
                    index=0,
                )
            ],
            created=1696507583,
            model="gpt-3.5-turbo-0613",
            object="chat.completion.chunk",
        ),
        ChatCompletionChunk(
            id="chatcmpl-86HPTGAKCDFXJwEM68ekpoBumMh4H",
            choices=[
                chat_completion_chunk.Choice(
                    delta=chat_completion_chunk.ChoiceDelta(
                        content=" test", function_call=None, role=None
                    ),
                    finish_reason="stop",
                    index=0,
                )
            ],
            created=1696507583,
            model="gpt-3.5-turbo-0613",
            object="chat.completion.chunk",
        ),
        ChatCompletionChunk(
            id="chatcmpl-86HPTGAKCDFXJwEM68ekpoBumMh4H",
            choices=[
                chat_completion_chunk.Choice(
                    delta=chat_completion_chunk.ChoiceDelta(
                        content=".", function_call=None, role=None
                    ),
                    finish_reason="stop",
                    index=0,
                )
            ],
            created=1696507583,
            model="gpt-3.5-turbo-0613",
            object="chat.completion.chunk",
        ),
        ChatCompletionChunk(
            id="chatcmpl-86HPTGAKCDFXJwEM68ekpoBumMh4H",
            choices=[
                chat_completion_chunk.Choice(
                    delta=chat_completion_chunk.ChoiceDelta(
                        content=None, function_call=None, role=None
                    ),
                    finish_reason="stop",
                    index=0,
                )
            ],
            created=1696507583,
            model="gpt-3.5-turbo-0613",
            object="chat.completion.chunk",
        ),
    ]


def test_openai_chat__stream(
    openai_chat_completion_stream: list[ChatCompletionChunk],
) -> None:
    with patch(
        "openai.resources.chat.completions.Completions.create"
    ) as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = (
                el for el in openai_chat_completion_stream
            )
            nebuly_init(observer=mock_observer)

            client = OpenAI(
                api_key="ciao",
            )
            for _ in client.chat.completions.create(  # type: ignore  # pylint: disable=not-an-iterable  # noqa: E501
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": "Say this is a test",
                    }
                ],
                stream=True,
                user_id="test_user",
                user_group_profile="test_group",
            ):
                ...
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


@pytest.fixture(name="openai_chat_completion_stream_async")
async def fixture_openai_chat_completion_stream_async(
    openai_chat_completion_stream: list[ChatCompletionChunk],
) -> AsyncGenerator[ChatCompletionChunk, None]:
    for el in openai_chat_completion_stream:
        yield el


@pytest.mark.asyncio
async def test_openai_chat__stream_async(
    openai_chat_completion_stream_async: AsyncGenerator[ChatCompletionChunk, None]
) -> None:
    with patch(
        "openai.resources.chat.completions.AsyncCompletions.create", new=AsyncMock()
    ) as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = openai_chat_completion_stream_async
            nebuly_init(observer=mock_observer)

            client = AsyncOpenAI(
                api_key="ciao",
            )
            async for _ in await client.chat.completions.create(  # type: ignore
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": "Say this is a test",
                    }
                ],
                stream=True,
                user_id="test_user",
                user_group_profile="test_group",
            ):
                ...
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )
