# pylint: disable=duplicate-code
from __future__ import annotations

import json
import sys
from typing import Any
from unittest.mock import patch

import langchain
import pytest
from langchain.agents import AgentExecutor, OpenAIFunctionsAgent, tool
from langchain.chains import LLMChain, SequentialChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import SystemMessage

from nebuly.contextmanager import new_interaction
from nebuly.entities import EventType, InteractionWatch, Observer, SpanWatch
from nebuly.observers import NebulyObserver
from nebuly.requests import CustomJSONEncoder
from tests import common

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 8, 1), reason="requires 3.8.1 or higher"
)


# Cache original functions
orig_func_llm_gen = langchain.llms.base.BaseLLM.generate
orig_func_chat_gen = langchain.chat_models.base.BaseChatModel.generate


def nebuly_init(observer: Observer) -> None:
    # Reset original functions
    langchain.llms.base.BaseLLM.generate = orig_func_llm_gen  # type: ignore
    langchain.chat_models.base.BaseChatModel.generate = orig_func_chat_gen  # type: ignore  # noqa: E501  # pylint: disable=line-too-long
    common.nebuly_init(observer)


@pytest.fixture(name="openai_completion")
def fixture_openai_completion() -> dict[str, Any]:
    return {
        "id": "cmpl-81JyWoIj5m9qz0M9g7aLBGtwZzUIg",
        "object": "text_completion",
        "created": 1695325804,
        "model": "gpt-3.5-turbo-instruct",
        "choices": [
            {
                "text": "Sample langchain response",
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11},
    }


def test_langchain_llm_chain__no_context_manager(
    openai_completion: dict[str, Any]
) -> None:
    with patch("openai.Completion.create") as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = openai_completion
            nebuly_init(mock_observer)
            llm = OpenAI(temperature=0.9)
            prompt = PromptTemplate(
                input_variables=["product"],
                template="What is a good name for a company that makes {product}?",
            )

            chain = LLMChain(llm=llm, prompt=prompt)
            result = chain.run(
                product="colorful socks",
                user_id="test_user",
                user_group_profile="test_group_profile",
            )

            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert (
                interaction_watch.input
                == "What is a good name for a company that makes colorful socks?"
            )
            assert interaction_watch.output == "Sample langchain response"
            assert interaction_watch.end_user == "test_user"
            assert interaction_watch.end_user_group_profile == "test_group_profile"
            assert len(interaction_watch.spans) == 3
            assert len(interaction_watch.hierarchy) == 3
            for span in interaction_watch.spans:
                assert isinstance(span, SpanWatch)
                assert span.provider_extras is not None
                if span.module == "langchain":
                    assert span.provider_extras.get("event_type") is not None
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


def test_langchain_llm_chain__with_context_manager(
    openai_completion: dict[str, Any]
) -> None:
    with patch("openai.Completion.create") as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = openai_completion
            nebuly_init(mock_observer)
            llm = OpenAI(temperature=0.9)
            prompt = PromptTemplate(
                input_variables=["product"],
                template="What is a good name for a company that makes {product}?",
            )

            chain = LLMChain(llm=llm, prompt=prompt)
            with new_interaction(
                user_id="test", user_group_profile="tier 1"
            ) as interaction:
                interaction.set_input("colorful socks new")
                result = chain.run("colorful socks")
                interaction.set_output("Sample langchain response")

            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "colorful socks new"
            assert interaction_watch.output == "Sample langchain response"
            assert interaction_watch.end_user == "test"
            assert interaction_watch.end_user_group_profile == "tier 1"
            assert len(interaction_watch.spans) == 3
            assert len(interaction_watch.hierarchy) == 3
            for span in interaction_watch.spans:
                assert isinstance(span, SpanWatch)
                assert span.provider_extras is not None
                if span.module == "langchain":
                    assert span.provider_extras.get("event_type") is not None
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


def test_langchain_llm_chain__multiple_chains_in_interaction(
    openai_completion: dict[str, Any],
) -> None:
    with patch("openai.Completion.create") as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = openai_completion
            nebuly_init(mock_observer)
            llm = OpenAI(temperature=0.9)
            prompt = PromptTemplate(
                input_variables=["product"],
                template="What is a good name for a company that makes {product}?",
            )

            chain = LLMChain(llm=llm, prompt=prompt)
            with new_interaction(
                user_id="test", user_group_profile="tier 1"
            ) as interaction:
                interaction.set_input("colorful socks new")
                result_0 = chain.run("colorful socks")
                result_1 = chain.run("colorful ties")
                interaction.set_output("Sample langchain response")

            assert result_0 is not None
            assert result_1 is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "colorful socks new"
            assert interaction_watch.output == "Sample langchain response"
            assert interaction_watch.end_user == "test"
            assert interaction_watch.end_user_group_profile == "tier 1"
            assert len(interaction_watch.spans) == 6
            assert len(interaction_watch.hierarchy) == 6
            for span in interaction_watch.spans:
                assert isinstance(span, SpanWatch)
                assert span.provider_extras is not None
                if span.module == "langchain":
                    assert span.provider_extras.get("event_type") is not None
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


def test_langchain_llm_chain__multiple_interactions(
    openai_completion: dict[str, Any]
) -> None:
    with patch("openai.Completion.create") as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = openai_completion
            nebuly_init(mock_observer)
            llm = OpenAI(temperature=0.9)
            prompt = PromptTemplate(
                input_variables=["product"],
                template="What is a good name for a company that makes {product}?",
            )

            chain = LLMChain(llm=llm, prompt=prompt)
            with new_interaction(
                user_id="test", user_group_profile="tier 1"
            ) as interaction:
                interaction.set_input("colorful socks")
                result_0 = chain.run("colorful socks")
                interaction.set_output("Sample langchain response 1")
            with new_interaction(
                user_id="test", user_group_profile="tier 1"
            ) as interaction:
                interaction.set_input("colorful ties")
                result_1 = chain.run("colorful ties")
                interaction.set_output("Sample langchain response 2")

            assert result_0 is not None
            assert result_1 is not None
            assert mock_observer.call_count == 2
            interaction_watch_0 = mock_observer.call_args_list[0][0][0]
            assert isinstance(interaction_watch_0, InteractionWatch)
            assert interaction_watch_0.input == "colorful socks"
            assert interaction_watch_0.output == "Sample langchain response 1"
            assert interaction_watch_0.end_user == "test"
            assert interaction_watch_0.end_user_group_profile == "tier 1"
            assert len(interaction_watch_0.spans) == 3
            assert len(interaction_watch_0.hierarchy) == 3
            for span in interaction_watch_0.spans:
                assert isinstance(span, SpanWatch)
            assert (
                json.dumps(interaction_watch_0.to_dict(), cls=CustomJSONEncoder)
                is not None
            )
            interaction_watch_1 = mock_observer.call_args_list[1][0][0]
            assert isinstance(interaction_watch_1, InteractionWatch)
            assert interaction_watch_1.input == "colorful ties"
            assert interaction_watch_1.output == "Sample langchain response 2"
            assert interaction_watch_1.end_user == "test"
            assert interaction_watch_1.end_user_group_profile == "tier 1"
            assert len(interaction_watch_1.spans) == 3
            assert len(interaction_watch_1.hierarchy) == 3
            for span in interaction_watch_1.spans:
                assert isinstance(span, SpanWatch)
                assert span.provider_extras is not None
                if span.module == "langchain":
                    assert span.provider_extras.get("event_type") is not None
            assert (
                json.dumps(interaction_watch_1.to_dict(), cls=CustomJSONEncoder)
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


def test_langchain_chat_chain__no_context_manager(openai_chat: dict[str, Any]) -> None:
    with patch("openai.ChatCompletion.create") as mock_chat_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_chat_completion_create.return_value = openai_chat
            nebuly_init(mock_observer)
            llm = ChatOpenAI()

            chat_prompt = ChatPromptTemplate.from_messages(
                messages=[
                    ("user", "Hello! I am {name}"),
                    ("assistant", "Hi there! How can I assist you today?"),
                    ("user", "I need help with my computer."),
                ]
            )
            chain = LLMChain(llm=llm, prompt=chat_prompt)
            result = chain.run(prompt=chat_prompt, name="Valerio")

            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "I need help with my computer."
            assert interaction_watch.history == [
                ("human", "Hello! I am Valerio"),
                ("ai", "Hi there! How can I assist you today?"),
            ]
            assert interaction_watch.output == "Hi there! How can I assist you today?"
            assert len(interaction_watch.spans) == 3
            assert len(interaction_watch.hierarchy) == 3
            for span in interaction_watch.spans:
                assert isinstance(span, SpanWatch)
                assert span.provider_extras is not None
                if span.module == "langchain":
                    assert span.provider_extras.get("event_type") is not None

            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


@pytest.fixture(name="openai_chat_with_function")
def fixture_openai_chat_with_function() -> dict[str, Any]:
    return {
        "id": "chatcmpl-82LLsqQsyEqBvTMvGUr9jgc7w3NfZ",
        "object": "chat.completion",
        "created": 1695569424,
        "model": "gpt-3.5-turbo-0613",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "function_call": {
                        "name": "get_word_length",
                        "arguments": '{\n  "word": "educa"\n}',
                    },
                },
                "finish_reason": "function_call",
            }
        ],
        "usage": {"prompt_tokens": 80, "completion_tokens": 17, "total_tokens": 97},
    }


def test_langchain__chain_with_function_tool(
    openai_chat_with_function: dict[str, Any], openai_chat: dict[str, Any]
) -> None:
    with patch("openai.ChatCompletion.create") as mock_chat_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_chat_completion_create.side_effect = [
                openai_chat_with_function,
                openai_chat,
            ]
            nebuly_init(mock_observer)

            llm = ChatOpenAI(temperature=0)

            @tool
            def get_word_length(word: str) -> int:
                """Returns the length of a word."""
                return len(word)

            tools = [get_word_length]

            system_message = SystemMessage(
                content="You are very powerful assistant, but bad at "
                "calculating lengths of words."
            )
            prompt = OpenAIFunctionsAgent.create_prompt(system_message=system_message)
            agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)  # type: ignore  # noqa: E501  # pylint: disable=line-too-long
            agent_executor = AgentExecutor(agent=agent, tools=tools)  # type: ignore
            agent_executor.run("how many letters in the word educa?")

            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "how many letters in the word educa?"
            assert interaction_watch.history == [
                (
                    "system",
                    "You are very powerful assistant, but bad at "
                    "calculating lengths of words.",
                )
            ]
            assert interaction_watch.output == "Hi there! How can I assist you today?"
            assert len(interaction_watch.spans) == 6
            assert len(interaction_watch.hierarchy) == 6
            for span in interaction_watch.spans:
                assert isinstance(span, SpanWatch)
                if span.module == "langchain":
                    assert span.provider_extras is not None
                    event_type = span.provider_extras.get("event_type")
                    assert event_type is not None
                    if event_type == EventType.TOOL.value:
                        assert span.rag_source == "get_word_length"

            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


def test_langchain_sequential_chain_single_input_var(
    openai_completion: dict[str, Any]
) -> None:
    with patch("openai.Completion.create") as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = openai_completion
            nebuly_init(mock_observer)
            llm = OpenAI(temperature=0.7)
            synopsis_template = """
            Title: {title}
            Playwright: This is a synopsis for the above play:"""
            synopsis_prompt_template = PromptTemplate(
                input_variables=["title"], template=synopsis_template
            )
            synopsis_chain = LLMChain(
                llm=llm, prompt=synopsis_prompt_template, output_key="synopsis"
            )
            llm = OpenAI(temperature=0.7)
            template = """
            Play Synopsis:
            {synopsis}
            Review from a New York Times play critic of the above play:"""
            prompt_template = PromptTemplate(
                input_variables=["synopsis"], template=template
            )
            review_chain = LLMChain(
                llm=llm, prompt=prompt_template, output_key="review"
            )

            overall_chain = SequentialChain(
                chains=[synopsis_chain, review_chain],
                input_variables=["title"],
                # Here we return multiple variables
                output_variables=["synopsis", "review"],
            )

            overall_chain(
                "Tragedy at sunset on the beach",
            )

            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert "Tragedy at sunset on the beach" in interaction_watch.input
            assert isinstance(interaction_watch.output, str)
            assert "review" in interaction_watch.output


def test_langchain_sequential_chain_multiple_input_vars(
    openai_completion: dict[str, Any]
) -> None:
    with patch("openai.Completion.create") as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = openai_completion
            nebuly_init(mock_observer)
            llm = OpenAI(temperature=0.7)
            synopsis_template = """
            Title: {title}
            Era: {era}
            Playwright: This is a synopsis for the above play:"""
            synopsis_prompt_template = PromptTemplate(
                input_variables=["title", "era"], template=synopsis_template
            )
            synopsis_chain = LLMChain(
                llm=llm, prompt=synopsis_prompt_template, output_key="synopsis"
            )
            llm = OpenAI(temperature=0.7)
            template = """
            Play Synopsis:
            {synopsis}
            Review from a New York Times play critic of the above play:"""
            prompt_template = PromptTemplate(
                input_variables=["synopsis"], template=template
            )
            review_chain = LLMChain(
                llm=llm, prompt=prompt_template, output_key="review"
            )

            overall_chain = SequentialChain(
                chains=[synopsis_chain, review_chain],
                input_variables=["era", "title"],
                # Here we return multiple variables
                output_variables=["synopsis", "review"],
            )

            overall_chain(
                {"title": "Tragedy at sunset on the beach", "era": "Victorian England"},
            )

            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert "Victorian England" in interaction_watch.input
            assert isinstance(interaction_watch.output, str)
            assert "review" in interaction_watch.output
