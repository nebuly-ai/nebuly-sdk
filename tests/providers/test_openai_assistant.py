from typing import List, Literal, Optional, Tuple
from unittest import mock

import pytest
from openai import AsyncOpenAI, OpenAI
from openai.pagination import AsyncCursorPage, SyncCursorPage
from openai.types.beta.threads import Message, Text, TextContentBlock

from nebuly.entities import HistoryEntry, InteractionWatch
from nebuly.utils import is_module_version_less_than
from tests.providers.common import nebuly_init

if is_module_version_less_than("openai", "1.21.0"):
    assistant_messages_list = (
        "openai.resources.beta.threads.messages.messages.Messages.list"
    )
    assistant_messages_async_list = (
        "openai.resources.beta.threads.messages.messages.AsyncMessages.list"
    )
else:
    assistant_messages_list = "openai.resources.beta.threads.messages.Messages.list"
    assistant_messages_async_list = (
        "openai.resources.beta.threads.messages.AsyncMessages.list"
    )


def create_fake_messages(
    messages: List[Tuple[Literal["user", "assistant"], str]],
    has_more: bool = False,
    reverse: bool = False,
) -> SyncCursorPage[Message]:
    """
    Create a fake messages list response
    """
    if reverse:
        messages = messages[::-1]
    response = SyncCursorPage(  # type: ignore[call-arg]
        data=[
            Message(
                id=f"message_{i}",
                assistant_id="assistant_1",
                status="completed",
                metadata=None,
                object="thread.message",
                role=role,
                run_id="run_1",
                thread_id="thread_1",
                content=[
                    TextContentBlock(
                        text=Text(value=message, annotations=[]),
                        type="text",
                    )
                ],
                created_at=1,
            )
            for i, (role, message) in enumerate(reversed(messages))
        ],
        has_more=has_more,
    )
    return response


def run_fake_assistant_messages(
    messages: List[Tuple[Literal["user", "assistant"], str]],
    has_more: bool = False,
    order: Optional[Literal["asc", "desc"]] = None,
) -> List[InteractionWatch]:
    """
    Mock the messages list from openai and run the assistant with the given
    messages
    """
    with mock.patch(
        assistant_messages_list,
    ) as mock_messages_list:
        reverse = order == "asc"
        response = create_fake_messages(messages, has_more=has_more, reverse=reverse)
        mock_messages_list.return_value = response

        dummy_observer: List[InteractionWatch] = []
        nebuly_init(dummy_observer.append)

        client = OpenAI(api_key="test_key")
        if order is not None:
            client.beta.threads.messages.list(  # type: ignore[call-arg]  # pylint: disable=unexpected-keyword-arg  # noqa: E501
                "thread_WPTJKw22r54NrYVM35FfHqYN", order=order, user="user_1"
            )
        else:
            client.beta.threads.messages.list(  # type: ignore[call-arg]  # pylint: disable=unexpected-keyword-arg  # noqa: E501
                "thread_WPTJKw22r54NrYVM35FfHqYN", user="user_1"
            )

    return dummy_observer


def create_fake_messages_async(
    messages: List[Tuple[Literal["user", "assistant"], str]],
    has_more: bool = False,
    reverse: bool = False,
) -> AsyncCursorPage[Message]:
    """
    Create a fake messages list response
    """
    if reverse:
        messages = messages[::-1]
    response = AsyncCursorPage(  # type: ignore[call-arg]
        data=[
            Message(
                id=f"message_{i}",
                assistant_id="assistant_1",
                metadata=None,
                status="completed",
                object="thread.message",
                role=role,
                run_id="run_1",
                thread_id="thread_1",
                content=[
                    TextContentBlock(
                        text=Text(value=message, annotations=[]),
                        type="text",
                    )
                ],
                created_at=1,
            )
            for i, (role, message) in enumerate(reversed(messages))
        ],
        has_more=has_more,
    )
    return response


async def run_fake_assistant_messages_async(
    messages: List[Tuple[Literal["user", "assistant"], str]],
    has_more: bool = False,
    order: Optional[Literal["asc", "desc"]] = None,
) -> List[InteractionWatch]:
    """
    Mock the messages list from openai and run the assistant with the given
    messages
    """
    with mock.patch(
        assistant_messages_async_list,
    ) as mock_messages_list:
        reverse = order == "asc"
        response = create_fake_messages_async(
            messages, has_more=has_more, reverse=reverse
        )
        mock_messages_list.return_value = response

        dummy_observer: List[InteractionWatch] = []
        nebuly_init(dummy_observer.append)

        client = AsyncOpenAI(api_key="test_key")
        if order is not None:
            await client.beta.threads.messages.list(  # type: ignore[call-arg]  # pylint: disable=unexpected-keyword-arg  # noqa: E501
                "thread_WPTJKw22r54NrYVM35FfHqYN", order=order, user="user_1"
            )
        else:
            await client.beta.threads.messages.list(  # type: ignore[call-arg]  # pylint: disable=unexpected-keyword-arg  # noqa: E501
                "thread_WPTJKw22r54NrYVM35FfHqYN", user="user_1"
            )

    return dummy_observer


def test_openai_assistant_messages_list() -> None:
    """
    Test that we correctly watch the messages list and send the correct
    interaction to the observer
    """
    dummy_observer = run_fake_assistant_messages(
        [
            ("user", "Hello"),
            ("assistant", "How can i help you?"),
            ("user", "You can't"),
            ("assistant", "Ok then"),
        ]
    )
    assert len(dummy_observer) == 1
    interaction_watch = dummy_observer[0]
    assert interaction_watch.input == "You can't"
    assert interaction_watch.output == "Ok then"
    assert interaction_watch.history == [HistoryEntry("Hello", "How can i help you?")]
    span = interaction_watch.spans[0]
    assert "thread_WPTJKw22r54NrYVM35FfHqYN" in span.called_with_args


def test_openai_assistant_messages_list_order_desc() -> None:
    """
    Test that we correctly watch the messages when the order is explicitly set
    to desc
    """
    dummy_observer = run_fake_assistant_messages(
        [
            ("user", "Hello"),
            ("assistant", "How can i help you?"),
            ("user", "You can't"),
            ("assistant", "Ok then"),
        ],
        order="desc",
    )
    assert len(dummy_observer) == 1
    interaction_watch = dummy_observer[0]
    assert interaction_watch.input == "You can't"
    assert interaction_watch.output == "Ok then"
    assert interaction_watch.history == [HistoryEntry("Hello", "How can i help you?")]
    span = interaction_watch.spans[0]
    assert "thread_WPTJKw22r54NrYVM35FfHqYN" in span.called_with_args


def test_openai_assistant_messages_list_order_asc() -> None:
    """
    Test that we correctly watch the messages when the order is explicitly set
    to asc
    """
    dummy_observer = run_fake_assistant_messages(
        [
            ("user", "Hello"),
            ("assistant", "How can i help you?"),
            ("user", "You can't"),
            ("assistant", "Ok then"),
        ],
        order="asc",
    )
    assert len(dummy_observer) == 1
    interaction_watch = dummy_observer[0]
    assert interaction_watch.input == "You can't"
    assert interaction_watch.output == "Ok then"
    assert interaction_watch.history == [HistoryEntry("Hello", "How can i help you?")]
    span = interaction_watch.spans[0]
    assert "thread_WPTJKw22r54NrYVM35FfHqYN" in span.called_with_args


def test_openai_assistant_messages_list_incomplete_response() -> None:
    """
    In the case when we dont get the full list of messages, we should only send
    the thread_id so the worker can fetch the rest of the messages
    """
    dummy_observer = run_fake_assistant_messages(
        [
            ("user", "Hello"),
            ("assistant", "How can i help you?"),
            ("user", "You can't"),
            ("assistant", "Ok then"),
        ],
        has_more=True,
    )
    assert len(dummy_observer) == 1
    interaction_watch = dummy_observer[0]
    assert interaction_watch.input == ""
    assert interaction_watch.output == ""
    assert interaction_watch.history == []
    assert len(interaction_watch.spans) == 1
    span = interaction_watch.spans[0]
    assert "thread_WPTJKw22r54NrYVM35FfHqYN" in span.called_with_args


def test_openai_assistant_messages_list_multiple_continuous_messages() -> None:
    """
    Test that we correctly watch the messages list when there are multiple
    continuous messages from the same role
    """
    dummy_observer = run_fake_assistant_messages(
        [
            ("user", "Hello"),
            ("user", "World"),
            ("assistant", "Hi"),
            ("assistant", "How can i help you?"),
            ("user", "I don't know"),
            ("user", "What can you do?"),
            ("assistant", "Not much"),
            ("assistant", "I'm just a bot"),
        ],
    )
    assert len(dummy_observer) == 1
    interaction_watch = dummy_observer[0]
    assert interaction_watch.input == "I don't know\nWhat can you do?"
    assert interaction_watch.output == "Not much\nI'm just a bot"
    assert interaction_watch.history == [
        HistoryEntry("Hello\nWorld", "Hi\nHow can i help you?"),
    ]
    assert len(interaction_watch.spans) == 1
    span = interaction_watch.spans[0]
    assert "thread_WPTJKw22r54NrYVM35FfHqYN" in span.called_with_args


def test_openai_assistant_messages_list_multiple_continuous_messages_order_asc() -> (
    None
):
    """
    Test that we correctly watch the messages list when there are multiple
    continuous messages from the same role when the order is explicitly set to
    asc
    """
    dummy_observer = run_fake_assistant_messages(
        [
            ("user", "Hello"),
            ("user", "World"),
            ("assistant", "Hi"),
            ("assistant", "How can i help you?"),
            ("user", "I don't know"),
            ("user", "What can you do?"),
            ("assistant", "Not much"),
            ("assistant", "I'm just a bot"),
        ],
        order="asc",
    )
    assert len(dummy_observer) == 1
    interaction_watch = dummy_observer[0]
    assert interaction_watch.input == "I don't know\nWhat can you do?"
    assert interaction_watch.output == "Not much\nI'm just a bot"
    assert interaction_watch.history == [
        HistoryEntry("Hello\nWorld", "Hi\nHow can i help you?"),
    ]
    assert len(interaction_watch.spans) == 1
    span = interaction_watch.spans[0]
    assert "thread_WPTJKw22r54NrYVM35FfHqYN" in span.called_with_args


@pytest.mark.asyncio
async def test_openai_assistant_messages_list_async() -> None:
    """
    Test that we correctly watch the messages list and send the correct
    interaction to the observer
    """
    dummy_observer = await run_fake_assistant_messages_async(
        [
            ("user", "Hello"),
            ("assistant", "How can i help you?"),
            ("user", "You can't"),
            ("assistant", "Ok then"),
        ]
    )
    assert len(dummy_observer) == 1
    interaction_watch = dummy_observer[0]
    assert interaction_watch.input == "You can't"
    assert interaction_watch.output == "Ok then"
    assert interaction_watch.history == [HistoryEntry("Hello", "How can i help you?")]
    span = interaction_watch.spans[0]
    assert "thread_WPTJKw22r54NrYVM35FfHqYN" in span.called_with_args
