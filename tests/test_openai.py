from typing import Any
from unittest.mock import patch

import openai
import pytest

import nebuly
from nebuly.contextmanager import new_interaction
from nebuly.entities import InteractionWatch, SpanWatch
from nebuly.observers import NebulyObserver


@pytest.fixture()
def openai_completion() -> dict:
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


def test_openai_completion__no_context_manager(openai_completion):
    with patch("openai.Completion.create") as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = openai_completion
            nebuly.init(api_key="test", disable_checks=True)
            result = openai.Completion.create(
                model="gpt-3.5-turbo-instruct",
                prompt="Say this is a test",
                max_tokens=7,
                temperature=0,
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


def test_openai_completion__with_context_manager(openai_completion):
    with patch("openai.Completion.create") as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = openai_completion
            nebuly.init(api_key="test", disable_checks=True)
            with new_interaction(
                user="test_user", group_profile="test_group"
            ) as interaction:
                interaction.set_input("Another input")
                result = openai.Completion.create(
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


def test_openai_completion__multiple_spans_in_interaction(openai_completion):
    with patch("openai.Completion.create") as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = openai_completion
            nebuly.init(api_key="test", disable_checks=True)
            with new_interaction(
                user="test_user", group_profile="test_group"
            ) as interaction:
                interaction.set_input("Initial input")
                result = openai.Completion.create(
                    model="gpt-3.5-turbo-instruct",
                    prompt="Say this is a test",
                    max_tokens=7,
                    temperature=0,
                )
                assert result is not None
                result = openai.Completion.create(
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


def test_openai_completion__multiple_interactions(openai_completion):
    with patch("openai.Completion.create") as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = openai_completion
            nebuly.init(api_key="test", disable_checks=True)
            with new_interaction(
                user="test_user", group_profile="test_group"
            ) as interaction:
                interaction.set_input("Initial input 1")
                result = openai.Completion.create(
                    model="gpt-3.5-turbo-instruct",
                    prompt="Say this is a test",
                    max_tokens=7,
                    temperature=0,
                )
                assert result is not None
                interaction.set_output("Final output 1")

            with new_interaction(
                user="test_user", group_profile="test_group"
            ) as interaction:
                interaction.set_input("Initial input 2")
                result = openai.Completion.create(
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


@pytest.mark.asyncio
async def test_openai_completion__async(openai_completion):
    with patch("openai.Completion.acreate") as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = openai_completion
            nebuly.init(api_key="test", disable_checks=True)
            result = await openai.Completion.acreate(
                model="gpt-3.5-turbo-instruct",
                prompt="Say this is a test",
                max_tokens=7,
                temperature=0,
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


@pytest.fixture()
def openai_chat() -> dict:
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


def test_openai_chat__no_context_manager(openai_chat):
    with patch("openai.ChatCompletion.create") as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = openai_chat
            nebuly.init(api_key="test", disable_checks=True)
            result = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello!"},
                ],
            )
            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "Hello!"
            assert interaction_watch.history == [
                ("system", "You are a helpful assistant."),
            ]
            assert interaction_watch.output == "Hi there! How can I assist you today?"
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)


def test_openai_chat__with_context_manager(openai_chat):
    with patch("openai.ChatCompletion.create") as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = openai_chat
            nebuly.init(api_key="test", disable_checks=True)
            with new_interaction(
                user="test_user", group_profile="test_group"
            ) as interaction:
                interaction.set_input("Hello!")
                interaction.set_history([("system", "You are a helpful assistant.")])
                result = openai.ChatCompletion.create(
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
            assert interaction_watch.history == [
                ("system", "You are a helpful assistant."),
            ]
            assert interaction_watch.output == "Another output"
            assert interaction.user == "test_user"
            assert interaction.user_group_profile == "test_group"
            assert len(interaction_watch.spans) == 1
            span: SpanWatch = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)


def test_openai_chat__multiple_spans_in_interaction(openai_chat):
    with patch("openai.ChatCompletion.create") as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = openai_chat
            nebuly.init(api_key="test", disable_checks=True)
            with new_interaction(
                user="test_user", group_profile="test_group"
            ) as interaction:
                interaction.set_input("Hello!")
                interaction.set_history([("system", "You are a helpful assistant.")])
                result = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hello!"},
                    ],
                )
                assert result is not None
                result = openai.ChatCompletion.create(
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
            assert interaction_watch.history == [
                ("system", "You are a helpful assistant."),
            ]
            assert interaction_watch.output == "Another output"
            assert interaction.user == "test_user"
            assert interaction.user_group_profile == "test_group"
            assert len(interaction_watch.spans) == 2
            span_0: SpanWatch = interaction_watch.spans[0]
            assert isinstance(span_0, SpanWatch)
            span_1: SpanWatch = interaction_watch.spans[1]
            assert isinstance(span_1, SpanWatch)
            assert span_0.span_id != span_1.span_id


def test_openai_chat__multiple_interactions(openai_chat):
    with patch("openai.ChatCompletion.create") as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = openai_chat
            nebuly.init(api_key="test", disable_checks=True)
            with new_interaction(
                user="test_user", group_profile="test_group"
            ) as interaction:
                interaction.set_input("Hello!")
                interaction.set_history([("system", "You are a helpful assistant.")])
                result = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hello!"},
                    ],
                )
                assert result is not None
                interaction.set_output("Another output")

            with new_interaction(
                user="test_user", group_profile="test_group"
            ) as interaction:
                interaction.set_input("Hello!")
                interaction.set_history([("system", "You are a helpful assistant.")])
                result = openai.ChatCompletion.create(
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
            assert interaction_watch_0.history == [
                ("system", "You are a helpful assistant."),
            ]
            assert interaction_watch_0.output == "Another output"
            assert interaction_watch_0.end_user == "test_user"
            assert interaction_watch_0.end_user_group_profile == "test_group"
            assert len(interaction_watch_0.spans) == 1
            span_0: SpanWatch = interaction_watch_0.spans[0]
            assert isinstance(span_0, SpanWatch)

            interaction_watch_1: InteractionWatch = mock_observer.call_args_list[1][0][
                0
            ]
            assert isinstance(interaction_watch_1, InteractionWatch)
            assert interaction_watch_1.input == "Hello!"
            assert interaction_watch_1.history == [
                ("system", "You are a helpful assistant."),
            ]
            assert interaction_watch_1.output == "Another output"
            assert interaction_watch_1.end_user == "test_user"
            assert interaction_watch_1.end_user_group_profile == "test_group"
            assert len(interaction_watch_1.spans) == 1
            span_1: SpanWatch = interaction_watch_1.spans[0]
            assert isinstance(span_1, SpanWatch)
            assert span_0.span_id != span_1.span_id


@pytest.mark.asyncio
async def test_openai_chat__async(openai_chat):
    with patch("openai.ChatCompletion.acreate") as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = openai_chat
            nebuly.init(api_key="test", disable_checks=True)
            result = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello!"},
                ],
            )
            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "Hello!"
            assert interaction_watch.history == [
                ("system", "You are a helpful assistant."),
            ]
            assert interaction_watch.output == "Hi there! How can I assist you today?"
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)


@pytest.fixture()
def openai_completion_gen() -> list[dict[str, Any]]:
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


def test_openai_completion_gen(openai_completion_gen):
    with patch("openai.Completion.create") as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = (el for el in openai_completion_gen)
            nebuly.init(api_key="test", disable_checks=True)
            result = ""
            for chunk in openai.Completion.create(
                model="gpt-3.5-turbo-instruct",
                prompt="Say this is a test",
                max_tokens=7,
                temperature=0,
                stream=True,
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


@pytest.fixture()
def openai_chat_gen() -> list[dict[str, Any]]:
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
                    "delta": {
                        "content": "",
                    },
                    "finish_reason": "stop",
                }
            ],
        },
    ]


def test_openai_chat_gen(openai_chat_gen):
    with patch("openai.ChatCompletion.create") as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = (el for el in openai_chat_gen)
            nebuly.init(api_key="test", disable_checks=True)
            result = ""
            for chunk in openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello!"},
                ],
                stream=True,
            ):
                result += chunk["choices"][0]["delta"]["content"]

            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "Hello!"
            assert interaction_watch.history == [
                ("system", "You are a helpful assistant."),
            ]
            assert interaction_watch.output == "Hello there"
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
