# pylint: disable=duplicate-code, import-error, no-name-in-module, wrong-import-position, wrong-import-order, too-many-lines  # noqa: E501
# mypy: ignore-errors
from __future__ import annotations

import json
import os
from importlib.metadata import version
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

if version("openai").startswith("0."):
    pytest.skip(
        "Tests enabled only when openai version is 1.X", allow_module_level=True
    )

import openai
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI
from openai._response import APIResponse
from openai.types import Completion, CompletionChoice, CompletionUsage
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    chat_completion_chunk,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import FunctionCall

from nebuly.contextmanager import new_interaction
from nebuly.entities import HistoryEntry, InteractionWatch, SpanWatch
from nebuly.observers import NebulyObserver
from nebuly.requests import CustomJSONEncoder
from tests.providers.common import nebuly_init

try:
    from openai._legacy_response import (  # pylint: disable=ungrouped-imports
        LegacyAPIResponse,
    )
except ImportError:
    LegacyAPIResponse = MagicMock

os.environ["OPENAI_API_KEY"] = "test_key"


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

            result = openai.completions.create(  # type: ignore  # pylint: disable=no-member  # noqa: E501
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


@pytest.fixture(name="openai_chat_completion_function_call")
def fixture_openai_chat_completion_function_call() -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-88VfyByOf2FT704n7rdk6lxa8p6XV",
        choices=[
            Choice(
                finish_reason="function_call",
                index=0,
                message=ChatCompletionMessage(
                    content=None,
                    role="assistant",
                    function_call=FunctionCall(
                        arguments='{\n  "location": "Boston"\n}',
                        name="get_current_weather",
                    ),
                ),
            )
        ],
        created=1697039078,
        model="gpt-3.5-turbo-0613",
        object="chat.completion",
        usage=CompletionUsage(completion_tokens=16, prompt_tokens=83, total_tokens=99),
    )


def test_openai_chat__function_call(
    openai_chat_completion_function_call: ChatCompletion,
) -> None:
    with patch(
        "openai.resources.chat.completions.Completions.create"
    ) as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = openai_chat_completion_function_call
            nebuly_init(observer=mock_observer)

            client = OpenAI(
                api_key="ciao",
            )
            functions = [
                {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. "
                                "San Francisco, CA",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location"],
                    },
                }
            ]
            messages = [
                {"role": "user", "content": "What's the weather like in Boston today?"}
            ]

            result = client.chat.completions.create(  # type: ignore
                model="gpt-3.5-turbo",
                messages=messages,
                functions=functions,
                function_call="auto",
                user_id="test_user",
                user_group_profile="test_group",
            )
            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "What's the weather like in Boston today?"
            assert interaction_watch.output == json.dumps(
                {
                    "function_name": "get_current_weather",
                    "arguments": '{\n  "location": "Boston"\n}',
                }
            )
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

            result = openai.chat.completions.create(  # type: ignore  # pylint: disable=no-member  # noqa: E501
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
                        "role": "system",
                        "content": "Say this is a test",
                    },
                    {
                        "role": "user",
                        "content": "Say this is a test",
                    },
                    {
                        "role": "assistant",
                        "content": "This is a test",
                    },
                    {
                        "role": "user",
                        "content": "Say again this is a test",
                    },
                ],
                user_id="test_user",
                user_group_profile="test_group",
                nebuly_tags={
                    "tenant": "ciao",
                },
                feature_flags=["flag1"],
            )
            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "Say again this is a test"
            assert interaction_watch.output == "\n\nThis is a test."
            assert interaction_watch.history == [
                HistoryEntry(user="Say this is a test", assistant="This is a test")
            ]
            assert interaction_watch.end_user == "test_user"
            assert interaction_watch.end_user_group_profile == "test_group"
            assert interaction_watch.tags == {"tenant": "ciao"}
            assert interaction_watch.feature_flags == ["flag1"]
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


@pytest.fixture(name="openai_chat_raw_response_legacy")
def fixture_openai_chat_raw_response_legacy() -> LegacyAPIResponse:
    mock_response = MagicMock(spec=LegacyAPIResponse)

    # set mock_response content
    mock_response.content = (
        b'{\n  "id": "chatcmpl-8wUgmL0M2b52YUg7lUUDfUWPp9JeP",\n  '
        b'"object": "chat.completion",\n  "created": 1708951684,'
        b'\n  "model": "gpt-3.5-turbo-0125",\n  "choices": [\n    '
        b'{\n      "index": 0,\n      "message": {\n        '
        b'"role": "assistant",\n        "content": "This is a '
        b'test."\n      },\n      "logprobs": '
        b'null,\n      "finish_reason": "stop"\n    }\n  ],\n  '
        b'"usage": {\n    "prompt_tokens": 19,\n    '
        b'"completion_tokens": 9,\n    "total_tokens": 28\n  '
        b'},\n  "system_fingerprint": "fp_86156a94a0"\n}\n'
    )

    return mock_response


def test_openai_chat__raw_response_legacy(
    openai_chat_raw_response_legacy: LegacyAPIResponse,
) -> None:
    with patch(
        "openai.resources.chat.completions.Completions.create"
    ) as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = (
                openai_chat_raw_response_legacy  # noqa: E501
            )
            nebuly_init(observer=mock_observer)

            client = OpenAI(
                api_key="ciao",
            )
            result = client.chat.completions.with_raw_response.create(  # type: ignore
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "Say this is a test",
                    },
                    {
                        "role": "user",
                        "content": "Say this is a test",
                    },
                    {
                        "role": "assistant",
                        "content": "This is a test",
                    },
                    {
                        "role": "user",
                        "content": "Say again this is a test",
                    },
                ],
                user_id="test_user",
                user_group_profile="test_group",
                nebuly_tags={
                    "tenant": "ciao",
                },
                feature_flags=["flag1"],
            )
            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "Say again this is a test"
            assert interaction_watch.output == "This is a test."
            assert interaction_watch.history == [
                HistoryEntry(user="Say this is a test", assistant="This is a test")
            ]
            assert interaction_watch.end_user == "test_user"
            assert interaction_watch.end_user_group_profile == "test_group"
            assert interaction_watch.tags == {"tenant": "ciao"}
            assert interaction_watch.feature_flags == ["flag1"]
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


@pytest.fixture(name="openai_chat_raw_response")
def fixture_openai_chat_raw_response() -> APIResponse:
    mock_response = MagicMock(spec=APIResponse)

    # set mock_response content
    mock_response.content = (
        b'{\n  "id": "chatcmpl-8wUgmL0M2b52YUg7lUUDfUWPp9JeP",\n  '
        b'"object": "chat.completion",\n  "created": 1708951684,'
        b'\n  "model": "gpt-3.5-turbo-0125",\n  "choices": [\n    '
        b'{\n      "index": 0,\n      "message": {\n        '
        b'"role": "assistant",\n        "content": "This is a '
        b'test."\n      },\n      "logprobs": '
        b'null,\n      "finish_reason": "stop"\n    }\n  ],\n  '
        b'"usage": {\n    "prompt_tokens": 19,\n    '
        b'"completion_tokens": 9,\n    "total_tokens": 28\n  '
        b'},\n  "system_fingerprint": "fp_86156a94a0"\n}\n'
    )

    return mock_response


def test_openai_chat__raw_response(
    openai_chat_raw_response: APIResponse,
) -> None:
    with patch(
        "openai.resources.chat.completions.Completions.create"
    ) as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = openai_chat_raw_response
            nebuly_init(observer=mock_observer)

            client = OpenAI(
                api_key="ciao",
            )
            result = client.chat.completions.with_raw_response.create(  # type: ignore
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "Say this is a test",
                    },
                    {
                        "role": "user",
                        "content": "Say this is a test",
                    },
                    {
                        "role": "assistant",
                        "content": "This is a test",
                    },
                    {
                        "role": "user",
                        "content": "Say again this is a test",
                    },
                ],
                user_id="test_user",
                user_group_profile="test_group",
                nebuly_tags={
                    "tenant": "ciao",
                },
                feature_flags=["flag1"],
            )
            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "Say again this is a test"
            assert interaction_watch.output == "This is a test."
            assert interaction_watch.history == [
                HistoryEntry(user="Say this is a test", assistant="This is a test")
            ]
            assert interaction_watch.end_user == "test_user"
            assert interaction_watch.end_user_group_profile == "test_group"
            assert interaction_watch.tags == {"tenant": "ciao"}
            assert interaction_watch.feature_flags == ["flag1"]
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


def test_azure_openai_chat__no_context_manager(
    openai_chat_completion: ChatCompletion,
) -> None:
    with patch(
        "openai.resources.chat.completions.Completions.create"
    ) as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = openai_chat_completion
            nebuly_init(observer=mock_observer)

            client = AzureOpenAI(
                api_version="2023-07-01-preview",
                azure_endpoint="https://testaisweden.openai.azure.com/",
                api_key="123456",
            )
            result = client.chat.completions.create(  # type: ignore
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "Say this is a test",
                    },
                    {
                        "role": "user",
                        "content": "Say this is a test",
                    },
                    {
                        "role": "assistant",
                        "content": "This is a test",
                    },
                    {
                        "role": "user",
                        "content": "Say again this is a test",
                    },
                ],
                user_id="test_user",
                user_group_profile="test_group",
                nebuly_tags={
                    "tenant": "ciao",
                },
                feature_flags=["flag1"],
            )
            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "Say again this is a test"
            assert interaction_watch.output == "\n\nThis is a test."
            assert interaction_watch.history == [
                HistoryEntry(user="Say this is a test", assistant="This is a test")
            ]
            assert interaction_watch.end_user == "test_user"
            assert interaction_watch.end_user_group_profile == "test_group"
            assert interaction_watch.tags == {"tenant": "ciao"}
            assert interaction_watch.feature_flags == ["flag1"]
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
                user_id="test_user",
                user_group_profile="test_group",
                nebuly_tags={"tenant": "ciao"},
                feature_flags=["flag1"],
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
            assert interaction_watch.tags == {"tenant": "ciao"}
            assert interaction_watch.feature_flags == ["flag1"]
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


@pytest.mark.asyncio
async def test_azure_openai_chat__async(openai_chat_completion: ChatCompletion) -> None:
    with patch(
        "openai.resources.chat.completions.AsyncCompletions.create", new=AsyncMock()
    ) as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = openai_chat_completion
            nebuly_init(observer=mock_observer)

            client = AsyncAzureOpenAI(
                api_version="2023-07-01-preview",
                azure_endpoint="https://testaisweden.openai.azure.com/",
                api_key="123456",
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


@pytest.fixture(name="openai_chat_completion_stream_function_call")
def fixture_openai_chat_completion_stream_function_call() -> list[ChatCompletionChunk]:
    return [
        ChatCompletionChunk(
            id="chatcmpl-88VsZy9x6lx55LPte3y8o65XBCMxL",
            choices=[
                chat_completion_chunk.Choice(
                    delta=chat_completion_chunk.ChoiceDelta(
                        content=None,
                        function_call=chat_completion_chunk.ChoiceDeltaFunctionCall(
                            arguments="", name="get_current_weather"
                        ),
                        role="assistant",
                    ),
                    finish_reason=None,
                    index=0,
                )
            ],
            created=1697039859,
            model="gpt-3.5-turbo-0613",
            object="chat.completion.chunk",
        ),
        ChatCompletionChunk(
            id="chatcmpl-88VsZy9x6lx55LPte3y8o65XBCMxL",
            choices=[
                chat_completion_chunk.Choice(
                    delta=chat_completion_chunk.ChoiceDelta(
                        content=None,
                        function_call=chat_completion_chunk.ChoiceDeltaFunctionCall(
                            arguments="{\n", name=None
                        ),
                        role=None,
                    ),
                    finish_reason=None,
                    index=0,
                )
            ],
            created=1697039859,
            model="gpt-3.5-turbo-0613",
            object="chat.completion.chunk",
        ),
        ChatCompletionChunk(
            id="chatcmpl-88VsZy9x6lx55LPte3y8o65XBCMxL",
            choices=[
                chat_completion_chunk.Choice(
                    delta=chat_completion_chunk.ChoiceDelta(
                        content=None,
                        function_call=chat_completion_chunk.ChoiceDeltaFunctionCall(
                            arguments=" ", name=None
                        ),
                        role=None,
                    ),
                    finish_reason=None,
                    index=0,
                )
            ],
            created=1697039859,
            model="gpt-3.5-turbo-0613",
            object="chat.completion.chunk",
        ),
        ChatCompletionChunk(
            id="chatcmpl-88VsZy9x6lx55LPte3y8o65XBCMxL",
            choices=[
                chat_completion_chunk.Choice(
                    delta=chat_completion_chunk.ChoiceDelta(
                        content=None,
                        function_call=chat_completion_chunk.ChoiceDeltaFunctionCall(
                            arguments=' "', name=None
                        ),
                        role=None,
                    ),
                    finish_reason=None,
                    index=0,
                )
            ],
            created=1697039859,
            model="gpt-3.5-turbo-0613",
            object="chat.completion.chunk",
        ),
        ChatCompletionChunk(
            id="chatcmpl-88VsZy9x6lx55LPte3y8o65XBCMxL",
            choices=[
                chat_completion_chunk.Choice(
                    delta=chat_completion_chunk.ChoiceDelta(
                        content=None,
                        function_call=chat_completion_chunk.ChoiceDeltaFunctionCall(
                            arguments="location", name=None
                        ),
                        role=None,
                    ),
                    finish_reason=None,
                    index=0,
                )
            ],
            created=1697039859,
            model="gpt-3.5-turbo-0613",
            object="chat.completion.chunk",
        ),
        ChatCompletionChunk(
            id="chatcmpl-88VsZy9x6lx55LPte3y8o65XBCMxL",
            choices=[
                chat_completion_chunk.Choice(
                    delta=chat_completion_chunk.ChoiceDelta(
                        content=None,
                        function_call=chat_completion_chunk.ChoiceDeltaFunctionCall(
                            arguments='":', name=None
                        ),
                        role=None,
                    ),
                    finish_reason=None,
                    index=0,
                )
            ],
            created=1697039859,
            model="gpt-3.5-turbo-0613",
            object="chat.completion.chunk",
        ),
        ChatCompletionChunk(
            id="chatcmpl-88VsZy9x6lx55LPte3y8o65XBCMxL",
            choices=[
                chat_completion_chunk.Choice(
                    delta=chat_completion_chunk.ChoiceDelta(
                        content=None,
                        function_call=chat_completion_chunk.ChoiceDeltaFunctionCall(
                            arguments=' "', name=None
                        ),
                        role=None,
                    ),
                    finish_reason=None,
                    index=0,
                )
            ],
            created=1697039859,
            model="gpt-3.5-turbo-0613",
            object="chat.completion.chunk",
        ),
        ChatCompletionChunk(
            id="chatcmpl-88VsZy9x6lx55LPte3y8o65XBCMxL",
            choices=[
                chat_completion_chunk.Choice(
                    delta=chat_completion_chunk.ChoiceDelta(
                        content=None,
                        function_call=chat_completion_chunk.ChoiceDeltaFunctionCall(
                            arguments="Boston", name=None
                        ),
                        role=None,
                    ),
                    finish_reason=None,
                    index=0,
                )
            ],
            created=1697039859,
            model="gpt-3.5-turbo-0613",
            object="chat.completion.chunk",
        ),
        ChatCompletionChunk(
            id="chatcmpl-88VsZy9x6lx55LPte3y8o65XBCMxL",
            choices=[
                chat_completion_chunk.Choice(
                    delta=chat_completion_chunk.ChoiceDelta(
                        content=None,
                        function_call=chat_completion_chunk.ChoiceDeltaFunctionCall(
                            arguments=",", name=None
                        ),
                        role=None,
                    ),
                    finish_reason=None,
                    index=0,
                )
            ],
            created=1697039859,
            model="gpt-3.5-turbo-0613",
            object="chat.completion.chunk",
        ),
        ChatCompletionChunk(
            id="chatcmpl-88VsZy9x6lx55LPte3y8o65XBCMxL",
            choices=[
                chat_completion_chunk.Choice(
                    delta=chat_completion_chunk.ChoiceDelta(
                        content=None,
                        function_call=chat_completion_chunk.ChoiceDeltaFunctionCall(
                            arguments=" MA", name=None
                        ),
                        role=None,
                    ),
                    finish_reason=None,
                    index=0,
                )
            ],
            created=1697039859,
            model="gpt-3.5-turbo-0613",
            object="chat.completion.chunk",
        ),
        ChatCompletionChunk(
            id="chatcmpl-88VsZy9x6lx55LPte3y8o65XBCMxL",
            choices=[
                chat_completion_chunk.Choice(
                    delta=chat_completion_chunk.ChoiceDelta(
                        content=None,
                        function_call=chat_completion_chunk.ChoiceDeltaFunctionCall(
                            arguments='"\n', name=None
                        ),
                        role=None,
                    ),
                    finish_reason=None,
                    index=0,
                )
            ],
            created=1697039859,
            model="gpt-3.5-turbo-0613",
            object="chat.completion.chunk",
        ),
        ChatCompletionChunk(
            id="chatcmpl-88VsZy9x6lx55LPte3y8o65XBCMxL",
            choices=[
                chat_completion_chunk.Choice(
                    delta=chat_completion_chunk.ChoiceDelta(
                        content=None,
                        function_call=chat_completion_chunk.ChoiceDeltaFunctionCall(
                            arguments="}", name=None
                        ),
                        role=None,
                    ),
                    finish_reason=None,
                    index=0,
                )
            ],
            created=1697039859,
            model="gpt-3.5-turbo-0613",
            object="chat.completion.chunk",
        ),
        ChatCompletionChunk(
            id="chatcmpl-88VsZy9x6lx55LPte3y8o65XBCMxL",
            choices=[
                chat_completion_chunk.Choice(
                    delta=chat_completion_chunk.ChoiceDelta(
                        content=None, function_call=None, role=None
                    ),
                    finish_reason="function_call",
                    index=0,
                )
            ],
            created=1697039859,
            model="gpt-3.5-turbo-0613",
            object="chat.completion.chunk",
        ),
    ]


def test_openai_chat__stream_function_call(
    openai_chat_completion_stream_function_call: list[ChatCompletionChunk],
) -> None:
    with patch(
        "openai.resources.chat.completions.Completions.create"
    ) as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = (
                el for el in openai_chat_completion_stream_function_call
            )
            nebuly_init(observer=mock_observer)

            client = OpenAI(
                api_key="ciao",
            )

            functions = [
                {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. "
                                "San Francisco, CA",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location"],
                    },
                }
            ]
            messages = [
                {"role": "user", "content": "What's the weather like in Boston today?"}
            ]

            for _ in client.chat.completions.create(  # type: ignore  # pylint: disable=not-an-iterable  # noqa: E501
                model="gpt-3.5-turbo",
                messages=messages,
                functions=functions,
                function_call="auto",
                stream=True,
                user_id="test_user",
                user_group_profile="test_group",
            ):
                ...

            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "What's the weather like in Boston today?"
            assert interaction_watch.output == json.dumps(
                {
                    "function_name": "get_current_weather",
                    "arguments": '{\n  "location": "Boston, MA"\n}',
                }
            )
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
            choices=[],
            created=1696507583,
            model="gpt-3.5-turbo-0613",
            object="chat.completion.chunk",
        ),
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


def test_openai_vision_models__url(
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
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "What is in this image?",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "https://upload.wikimedia.org/image1",
                                },
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "https://upload.wikimedia.org/image2",
                                },
                            },
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": "A beautiful landscape",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "https://upload.wikimedia.org/image1",
                                },
                            },
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "What is in this image?",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "https://upload.wikimedia.org/image3",
                                },
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "https://upload.wikimedia.org/image4",
                                },
                            },
                        ],
                    },
                ],
                user_id="test_user",
                user_group_profile="test_group",
                nebuly_tags={
                    "tenant": "ciao",
                },
                feature_flags=["flag1"],
            )
            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "What is in this image?"
            assert interaction_watch.output == "\n\nThis is a test."
            assert interaction_watch.history == [
                HistoryEntry(
                    user="What is in this image?",
                    assistant="A beautiful landscape",
                )
            ]
            assert interaction_watch.end_user == "test_user"
            assert interaction_watch.end_user_group_profile == "test_group"
            assert interaction_watch.tags == {"tenant": "ciao"}
            assert interaction_watch.feature_flags == ["flag1"]
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
            assert len(span.media) == 2
            assert span.media[0] == "https://upload.wikimedia.org/image3"

            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


def test_openai_vision_models__base64(
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
            base64_image = "sample_base64_image"
            result = client.chat.completions.create(  # type: ignore
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "What is in this image?",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                },
                            },
                        ],
                    }
                ],
                user_id="test_user",
                user_group_profile="test_group",
                nebuly_tags={
                    "tenant": "ciao",
                },
                feature_flags=["flag1"],
            )
            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "What is in this image?"
            assert interaction_watch.output == "\n\nThis is a test."
            assert len(interaction_watch.history) == 0
            assert interaction_watch.end_user == "test_user"
            assert interaction_watch.end_user_group_profile == "test_group"
            assert interaction_watch.tags == {"tenant": "ciao"}
            assert interaction_watch.feature_flags == ["flag1"]
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
            assert len(span.media) == 1
            assert span.media[0] == f"data:image/jpeg;base64,{base64_image}"

            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


def test_openai_vision_models__no_media(
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
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "What is in this image?",
                            }
                        ],
                    }
                ],
                user_id="test_user",
                user_group_profile="test_group",
                nebuly_tags={
                    "tenant": "ciao",
                },
                feature_flags=["flag1"],
            )
            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "What is in this image?"
            assert interaction_watch.output == "\n\nThis is a test."
            assert len(interaction_watch.history) == 0
            assert interaction_watch.end_user == "test_user"
            assert interaction_watch.end_user_group_profile == "test_group"
            assert interaction_watch.tags == {"tenant": "ciao"}
            assert interaction_watch.feature_flags == ["flag1"]
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
            assert len(span.media) == 0
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )
