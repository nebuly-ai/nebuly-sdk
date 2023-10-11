# pylint: disable=duplicate-code, import-error, no-name-in-module, wrong-import-position, wrong-import-order, no-member  # noqa: E501
from __future__ import annotations

import json
from importlib.metadata import version
from typing import Any, AsyncGenerator
from unittest.mock import patch

import openai
import pytest

if not version("openai").startswith("0."):
    # pylint: disable=import-error, no-name-in-module
    pytest.skip("Legacy tests only when openai version is 0.X", allow_module_level=True)

from nebuly.contextmanager import new_interaction
from nebuly.entities import HistoryEntry, InteractionWatch, SpanWatch
from nebuly.observers import NebulyObserver
from nebuly.requests import CustomJSONEncoder
from tests.common import nebuly_init


@pytest.fixture(name="openai_completion")
def fixture_openai_completion() -> dict[str, Any]:
    return {
        "id": "cmpl-81JyWoIj5m9qz0M9g7aLBGtwZzUIg",
        "object": "text_completion",
        "created": 1695325804,
        "model": "gpt-3.5-turbo-instruct",
        "choices": [
            {
                "text": "\n\nThis is a test.",
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11},
    }


def test_openai_completion__no_context_manager(
    openai_completion: dict[str, Any]
) -> None:
    with patch("openai.Completion.create") as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = openai_completion
            nebuly_init(observer=mock_observer)
            result = openai.Completion.create(  # type: ignore
                model="gpt-3.5-turbo-instruct",
                prompt="Say this is a test",
                max_tokens=7,
                temperature=0,
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


def test_openai_completion__with_context_manager(
    openai_completion: dict[str, Any]
) -> None:
    with patch("openai.Completion.create") as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = openai_completion
            nebuly_init(observer=mock_observer)
            with new_interaction(
                user_id="test_user", user_group_profile="test_group"
            ) as interaction:
                interaction.set_input("Another input")
                result = openai.Completion.create(  # type: ignore
                    model="gpt-3.5-turbo-instruct",
                    prompt="Say this is a test",
                    max_tokens=7,
                    temperature=0,
                )
                assert result is not None
                interaction.set_output("Another output")

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
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


def test_openai_completion__multiple_spans_in_interaction(
    openai_completion: dict[str, Any]
) -> None:
    with patch("openai.Completion.create") as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = openai_completion
            nebuly_init(observer=mock_observer)
            with new_interaction(
                user_id="test_user", user_group_profile="test_group"
            ) as interaction:
                interaction.set_input("Initial input")
                result = openai.Completion.create(  # type: ignore
                    model="gpt-3.5-turbo-instruct",
                    prompt="Say this is a test",
                    max_tokens=7,
                    temperature=0,
                )
                assert result is not None
                result = openai.Completion.create(  # type: ignore
                    model="gpt-3.5-turbo-instruct",
                    prompt="Say this is a test",
                    max_tokens=7,
                    temperature=0,
                )
                assert result is not None
                interaction.set_output("Final output")

            assert mock_observer.call_count == 1
            interaction_watch: InteractionWatch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "Initial input"
            assert interaction_watch.output == "Final output"
            assert interaction.user == "test_user"
            assert interaction.user_group_profile == "test_group"
            assert len(interaction_watch.spans) == 2
            span_0: SpanWatch = interaction_watch.spans[0]
            assert isinstance(span_0, SpanWatch)
            span_1: SpanWatch = interaction_watch.spans[1]
            assert isinstance(span_1, SpanWatch)
            assert span_0.span_id != span_1.span_id
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


def test_openai_completion__multiple_interactions(
    openai_completion: dict[str, Any]
) -> None:
    with patch("openai.Completion.create") as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = openai_completion
            nebuly_init(observer=mock_observer)
            with new_interaction(
                user_id="test_user", user_group_profile="test_group"
            ) as interaction:
                interaction.set_input("Initial input 1")
                result = openai.Completion.create(  # type: ignore
                    model="gpt-3.5-turbo-instruct",
                    prompt="Say this is a test",
                    max_tokens=7,
                    temperature=0,
                )
                assert result is not None
                interaction.set_output("Final output 1")

            with new_interaction(
                user_id="test_user", user_group_profile="test_group"
            ) as interaction:
                interaction.set_input("Initial input 2")
                result = openai.Completion.create(  # type: ignore
                    model="gpt-3.5-turbo-instruct",
                    prompt="Say this is a test",
                    max_tokens=7,
                    temperature=0,
                )
                assert result is not None
                interaction.set_output("Final output 2")

            assert mock_observer.call_count == 2
            interaction_watch_0: InteractionWatch = mock_observer.call_args_list[0][0][
                0
            ]
            assert isinstance(interaction_watch_0, InteractionWatch)
            assert interaction_watch_0.input == "Initial input 1"
            assert interaction_watch_0.output == "Final output 1"
            assert interaction_watch_0.end_user == "test_user"
            assert interaction_watch_0.end_user_group_profile == "test_group"
            assert len(interaction_watch_0.spans) == 1
            span_0: SpanWatch = interaction_watch_0.spans[0]
            assert isinstance(span_0, SpanWatch)
            assert (
                json.dumps(interaction_watch_0.to_dict(), cls=CustomJSONEncoder)
                is not None
            )

            interaction_watch_1: InteractionWatch = mock_observer.call_args_list[1][0][
                0
            ]
            assert isinstance(interaction_watch_1, InteractionWatch)
            assert interaction_watch_1.input == "Initial input 2"
            assert interaction_watch_1.output == "Final output 2"
            assert interaction_watch_1.end_user == "test_user"
            assert interaction_watch_1.end_user_group_profile == "test_group"
            assert len(interaction_watch_1.spans) == 1
            span_1: SpanWatch = interaction_watch_1.spans[0]
            assert isinstance(span_1, SpanWatch)
            assert span_0.span_id != span_1.span_id
            assert (
                json.dumps(interaction_watch_1.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


@pytest.mark.asyncio
async def test_openai_completion__async(openai_completion: dict[str, Any]) -> None:
    with patch("openai.Completion.acreate") as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = openai_completion
            nebuly_init(observer=mock_observer)
            result = await openai.Completion.acreate(  # type: ignore
                model="gpt-3.5-turbo-instruct",
                prompt="Say this is a test",
                max_tokens=7,
                temperature=0,
                user_id="test_user",
                user_group_profile="test_group",
            )
            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "Say this is a test"
            assert interaction_watch.output == "\n\nThis is a test."
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


@pytest.fixture(name="openai_chat_function_call")
def fixture_openai_chat_function_call() -> dict[str, Any]:
    return {
        "id": "chatcmpl-81Kl80GyhDVsOiEBQLQ6vG8svCUPe",
        "object": "chat.completion",
        "created": 1695328818,
        "model": "gpt-3.5-turbo-0613",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "function_call": {
                        "name": "get_current_weather",
                        "arguments": '{\n"location": "Boston, MA"\n}',
                    },
                },
                "finish_reason": "function_call",
            }
        ],
        "usage": {"prompt_tokens": 19, "completion_tokens": 10, "total_tokens": 29},
    }


def test_openai_chat__function_call(openai_chat_function_call: dict[str, Any]) -> None:
    with patch("openai.ChatCompletion.create") as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = openai_chat_function_call
            nebuly_init(observer=mock_observer)
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

            result = openai.ChatCompletion.create(  # type: ignore
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
            assert interaction_watch.history == []
            assert interaction_watch.output == json.dumps(
                {
                    "function_name": "get_current_weather",
                    "arguments": '{\n"location": "Boston, MA"\n}',
                }
            )
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


@pytest.fixture(name="openai_chat")
def fixture_openai_chat() -> dict[str, Any]:
    return {
        "id": "chatcmpl-81Kl80GyhDVsOiEBQLQ6vG8svCUPe",
        "object": "chat.completion",
        "created": 1695328818,
        "model": "gpt-3.5-turbo-0613",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hi there! How can I assist you today?",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 19, "completion_tokens": 10, "total_tokens": 29},
    }


def test_openai_chat__no_context_manager(openai_chat: dict[str, Any]) -> None:
    with patch("openai.ChatCompletion.create") as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = openai_chat
            nebuly_init(observer=mock_observer)
            result = openai.ChatCompletion.create(  # type: ignore
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello!"},
                ],
                user_id="test_user",
                user_group_profile="test_group",
            )
            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "Hello!"
            assert interaction_watch.history == []
            assert interaction_watch.output == "Hi there! How can I assist you today?"
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


def test_openai_chat__with_context_manager(openai_chat: dict[str, Any]) -> None:
    with patch("openai.ChatCompletion.create") as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = openai_chat
            nebuly_init(observer=mock_observer)
            with new_interaction(
                user_id="test_user", user_group_profile="test_group"
            ) as interaction:
                interaction.set_input("Hello!")
                interaction.set_history([("system", "You are a helpful assistant.")])
                result = openai.ChatCompletion.create(  # type: ignore
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hello!"},
                    ],
                )
                assert result is not None
                interaction.set_output("Another output")

            assert mock_observer.call_count == 1
            interaction_watch: InteractionWatch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "Hello!"
            assert interaction_watch.history == []
            assert interaction_watch.output == "Another output"
            assert interaction.user == "test_user"
            assert interaction.user_group_profile == "test_group"
            assert len(interaction_watch.spans) == 1
            span: SpanWatch = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


def test_openai_chat__multiple_spans_in_interaction(
    openai_chat: dict[str, Any]
) -> None:
    with patch("openai.ChatCompletion.create") as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = openai_chat
            nebuly_init(observer=mock_observer)
            with new_interaction(
                user_id="test_user", user_group_profile="test_group"
            ) as interaction:
                interaction.set_input("Hello!")
                interaction.set_history([("system", "You are a helpful assistant.")])
                result = openai.ChatCompletion.create(  # type: ignore
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hello!"},
                    ],
                )
                assert result is not None
                result = openai.ChatCompletion.create(  # type: ignore
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hello!"},
                    ],
                )
                assert result is not None
                interaction.set_output("Another output")

            assert mock_observer.call_count == 1
            interaction_watch: InteractionWatch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "Hello!"
            assert interaction_watch.history == []
            assert interaction_watch.output == "Another output"
            assert interaction.user == "test_user"
            assert interaction.user_group_profile == "test_group"
            assert len(interaction_watch.spans) == 2
            span_0: SpanWatch = interaction_watch.spans[0]
            assert isinstance(span_0, SpanWatch)
            span_1: SpanWatch = interaction_watch.spans[1]
            assert isinstance(span_1, SpanWatch)
            assert span_0.span_id != span_1.span_id
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


def test_openai_chat__multiple_interactions(openai_chat: dict[str, Any]) -> None:
    with patch("openai.ChatCompletion.create") as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = openai_chat
            nebuly_init(observer=mock_observer)
            with new_interaction(
                user_id="test_user", user_group_profile="test_group"
            ) as interaction:
                interaction.set_input("Hello!")
                interaction.set_history([("system", "You are a helpful assistant.")])
                result = openai.ChatCompletion.create(  # type: ignore
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hello!"},
                    ],
                )
                assert result is not None
                interaction.set_output("Another output")

            with new_interaction(
                user_id="test_user", user_group_profile="test_group"
            ) as interaction:
                interaction.set_input("Hello!")
                interaction.set_history([("system", "You are a helpful assistant.")])
                result = openai.ChatCompletion.create(  # type: ignore
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hello!"},
                    ],
                )
                assert result is not None
                interaction.set_output("Another output")

            assert mock_observer.call_count == 2
            interaction_watch_0: InteractionWatch = mock_observer.call_args_list[0][0][
                0
            ]
            assert isinstance(interaction_watch_0, InteractionWatch)
            assert interaction_watch_0.input == "Hello!"
            assert interaction_watch_0.history == []
            assert interaction_watch_0.output == "Another output"
            assert interaction_watch_0.end_user == "test_user"
            assert interaction_watch_0.end_user_group_profile == "test_group"
            assert len(interaction_watch_0.spans) == 1
            assert (
                json.dumps(interaction_watch_0.to_dict(), cls=CustomJSONEncoder)
                is not None
            )
            span_0: SpanWatch = interaction_watch_0.spans[0]
            assert isinstance(span_0, SpanWatch)

            interaction_watch_1: InteractionWatch = mock_observer.call_args_list[1][0][
                0
            ]
            assert isinstance(interaction_watch_1, InteractionWatch)
            assert interaction_watch_1.input == "Hello!"
            assert interaction_watch_1.history == []
            assert interaction_watch_1.output == "Another output"
            assert interaction_watch_1.end_user == "test_user"
            assert interaction_watch_1.end_user_group_profile == "test_group"
            assert len(interaction_watch_1.spans) == 1
            span_1: SpanWatch = interaction_watch_1.spans[0]
            assert isinstance(span_1, SpanWatch)
            assert span_0.span_id != span_1.span_id
            assert (
                json.dumps(interaction_watch_1.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


@pytest.mark.asyncio
async def test_openai_chat__async(openai_chat: dict[str, Any]) -> None:
    with patch("openai.ChatCompletion.acreate") as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = openai_chat
            nebuly_init(observer=mock_observer)
            result = await openai.ChatCompletion.acreate(  # type: ignore
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": "Hello!"},
                    {"role": "assistant", "content": "Hello!"},
                    {"role": "user", "content": "Hello!"},
                ],
                user_id="test_user",
                user_group_profile="test_group",
            )
            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "Hello!"
            assert interaction_watch.history == [
                HistoryEntry(user="Hello!", assistant="Hello!")
            ]
            assert interaction_watch.output == "Hi there! How can I assist you today?"
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


@pytest.fixture(name="openai_completion_gen")
def fixture_openai_completion_gen() -> list[dict[str, Any]]:
    return [
        {
            "id": "cmpl-81Z9F6z257cbJK1ZB9LQTuG8ePppT",
            "object": "text_completion",
            "created": 1695384129,
            "choices": [
                {"text": "\n\n", "index": 0, "logprobs": None, "finish_reason": None}
            ],
            "model": "gpt-3.5-turbo-instruct",
        },
        {
            "id": "cmpl-81Z9F6z257cbJK1ZB9LQTuG8ePppT",
            "object": "text_completion",
            "created": 1695384129,
            "choices": [
                {"text": "This", "index": 0, "logprobs": None, "finish_reason": None}
            ],
            "model": "gpt-3.5-turbo-instruct",
        },
        {
            "id": "cmpl-81Z9F6z257cbJK1ZB9LQTuG8ePppT",
            "object": "text_completion",
            "created": 1695384129,
            "choices": [
                {"text": " is", "index": 0, "logprobs": None, "finish_reason": None}
            ],
            "model": "gpt-3.5-turbo-instruct",
        },
        {
            "id": "cmpl-81Z9F6z257cbJK1ZB9LQTuG8ePppT",
            "object": "text_completion",
            "created": 1695384129,
            "choices": [
                {"text": " a", "index": 0, "logprobs": None, "finish_reason": None}
            ],
            "model": "gpt-3.5-turbo-instruct",
        },
        {
            "id": "cmpl-81Z9F6z257cbJK1ZB9LQTuG8ePppT",
            "object": "text_completion",
            "created": 1695384129,
            "choices": [
                {"text": " test", "index": 0, "logprobs": None, "finish_reason": None}
            ],
            "model": "gpt-3.5-turbo-instruct",
        },
        {
            "id": "cmpl-81Z9F6z257cbJK1ZB9LQTuG8ePppT",
            "object": "text_completion",
            "created": 1695384129,
            "choices": [
                {"text": ".", "index": 0, "logprobs": None, "finish_reason": None}
            ],
            "model": "gpt-3.5-turbo-instruct",
        },
        {
            "id": "cmpl-81Z9F6z257cbJK1ZB9LQTuG8ePppT",
            "object": "text_completion",
            "created": 1695384129,
            "choices": [
                {"text": "", "index": 0, "logprobs": None, "finish_reason": "stop"}
            ],
            "model": "gpt-3.5-turbo-instruct",
        },
    ]


def test_openai_completion_gen(openai_completion_gen: list[dict[str, Any]]) -> None:
    with patch("openai.Completion.create") as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = (el for el in openai_completion_gen)
            nebuly_init(observer=mock_observer)
            result = ""
            for chunk in openai.Completion.create(  # type: ignore
                model="gpt-3.5-turbo-instruct",
                prompt="Say this is a test",
                max_tokens=7,
                temperature=0,
                stream=True,
                user_id="test_user",
                user_group_profile="test_group",
            ):
                result += chunk["choices"][0]["text"]

            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "Say this is a test"
            assert interaction_watch.output == "\n\nThis is a test."
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


@pytest.fixture(name="openai_completion_gen_async")
async def fixture_openai_completion_gen_async(
    openai_completion_gen: list[dict[str, Any]]
) -> AsyncGenerator[dict[str, Any], None]:
    for el in openai_completion_gen:
        yield el


@pytest.mark.asyncio
async def test_openai_completion_gen_async(
    openai_completion_gen_async: AsyncGenerator[dict[str, Any], None]
) -> None:
    with patch("openai.Completion.acreate") as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = openai_completion_gen_async
            nebuly_init(observer=mock_observer)
            async for _ in await openai.Completion.acreate(  # type: ignore
                model="gpt-3.5-turbo-instruct",
                prompt="Say this is a test",
                max_tokens=7,
                temperature=0,
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
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


@pytest.fixture(name="openai_chat_gen_function_call")
def fixture_openai_chat_gen_function_call() -> list[dict[str, Any]]:
    return [
        {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1677652288,
            "model": "gpt-3.5-turbo",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": None,
                        "function_call": {
                            "name": "get_current_weather",
                            "arguments": "",
                        },
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1677652288,
            "model": "gpt-3.5-turbo",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "function_call": {"arguments": '{\n"location": "Boston, MA"\n}'}
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1677652288,
            "model": "gpt-3.5-turbo",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "function_call"}],
        },
    ]


def test_openai_chat_gen_function_call(
    openai_chat_gen_function_call: list[dict[str, Any]]
) -> None:
    with patch("openai.ChatCompletion.create") as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = (
                el for el in openai_chat_gen_function_call
            )
            nebuly_init(observer=mock_observer)
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
            for _ in openai.ChatCompletion.create(  # type: ignore
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
            assert interaction_watch.history == []
            assert interaction_watch.output == json.dumps(
                {
                    "function_name": "get_current_weather",
                    "arguments": '{\n"location": "Boston, MA"\n}',
                }
            )
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


@pytest.fixture(name="openai_chat_gen")
def fixture_openai_chat_gen() -> list[dict[str, Any]]:
    return [
        {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1677652288,
            "model": "gpt-3.5-turbo",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": "Hello",
                    },
                    "finish_reason": "stop",
                }
            ],
        },
        {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1677652288,
            "model": "gpt-3.5-turbo",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": " there",
                    },
                    "finish_reason": "stop",
                }
            ],
        },
        {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1677652288,
            "model": "gpt-3.5-turbo",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        },
    ]


def test_openai_chat_gen(openai_chat_gen: list[dict[str, Any]]) -> None:
    with patch("openai.ChatCompletion.create") as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = (el for el in openai_chat_gen)
            nebuly_init(observer=mock_observer)
            result = ""
            for chunk in openai.ChatCompletion.create(  # type: ignore
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello!"},
                ],
                stream=True,
                user_id="test_user",
                user_group_profile="test_group",
            ):
                result += chunk["choices"][0]["delta"].get("content", "")

            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "Hello!"
            assert interaction_watch.history == []
            assert interaction_watch.output == "Hello there"
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


@pytest.fixture(name="openai_chat_gen_async")
async def fixture_openai_chat_gen_async(
    openai_chat_gen: list[dict[str, Any]]
) -> AsyncGenerator[dict[str, Any], None]:
    for el in openai_chat_gen:
        yield el


@pytest.mark.asyncio
async def test_openai_chat_gen_async(
    openai_chat_gen_async: AsyncGenerator[dict[str, Any], None]
) -> None:
    with patch("openai.ChatCompletion.acreate") as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = openai_chat_gen_async
            nebuly_init(observer=mock_observer)
            result = ""
            async for chunk in await openai.ChatCompletion.acreate(  # type: ignore
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello!"},
                ],
                stream=True,
                user_id="test_user",
                user_group_profile="test_group",
            ):
                result += chunk["choices"][0]["delta"].get("content", "")

            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "Hello!"
            assert interaction_watch.history == []
            assert interaction_watch.output == "Hello there"
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )
