# flake8: noqa: E501
# pylint: disable=duplicate-code, wrong-import-position,line-too-long,wrong-import-order,ungrouped-imports
from __future__ import annotations

import json
import sys
from importlib.metadata import version
from typing import Any
from unittest.mock import Mock, patch

import pytest

from nebuly.providers.langchain import LangChainTrackingHandler

if sys.version_info < (3, 8, 1):
    # pylint: disable=import-error, no-name-in-module
    pytest.skip("Cannot use langchain in python<3.8.1", allow_module_level=True)

if not version("openai").startswith("0."):
    pytest.skip("Langchain doesn't support openai 1.X", allow_module_level=True)

from langchain.agents import AgentExecutor, OpenAIFunctionsAgent, tool
from langchain.chains import LLMChain, SequentialChain
from langchain.chat_models import ChatOpenAI
from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import StrOutputParser, SystemMessage

from nebuly.entities import EventType, HistoryEntry, InteractionWatch, SpanWatch
from nebuly.requests import CustomJSONEncoder


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


def test_langchain_llm_chain__callback_on_model(
    openai_completion: dict[str, Any]
) -> None:
    with patch("openai.Completion.create") as mock_completion_create, patch(
        "nebuly.providers.langchain.post_message",
        return_value=Mock(),
    ) as mock_send_interaction:
        mock_completion_create.return_value = openai_completion
        callback = LangChainTrackingHandler(
            api_key="test_key",
            user_id="test_user",
        )
        llm = OpenAI(temperature=0.9, openai_api_key="test", callbacks=[callback])
        prompt = PromptTemplate(
            input_variables=["product"],
            template="What is a good name for a company that makes {product}?",
        )

        chain = LLMChain(llm=llm, prompt=prompt)
        result = chain.run(
            product="colorful socks",
        )

        assert result is not None
        assert mock_send_interaction.call_count == 1
        interaction_watch = mock_send_interaction.call_args[0][0]
        assert isinstance(interaction_watch, InteractionWatch)
        assert (
            interaction_watch.input
            == "What is a good name for a company that makes colorful socks?"
        )
        assert interaction_watch.output == "Sample langchain response"
        assert interaction_watch.end_user == "test_user"
        assert len(interaction_watch.spans) == 1
        assert len(interaction_watch.hierarchy) == 1
        for span in interaction_watch.spans:
            assert isinstance(span, SpanWatch)
            assert span.provider_extras is not None
            if span.module == "langchain":
                assert span.provider_extras.get("event_type") is not None
        assert (
            json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder) is not None
        )


def test_langchain_llm_chain__callback_on_chain(
    openai_completion: dict[str, Any]
) -> None:
    with patch("openai.Completion.create") as mock_completion_create, patch(
        "nebuly.providers.langchain.post_message",
        return_value=Mock(),
    ) as mock_send_interaction:
        mock_completion_create.return_value = openai_completion
        callback = LangChainTrackingHandler(
            api_key="test_key",
            user_id="test_user",
        )
        llm = OpenAI(temperature=0.9, openai_api_key="test")
        prompt = PromptTemplate(
            input_variables=["product"],
            template="What is a good name for a company that makes {product}?",
        )

        chain = LLMChain(llm=llm, prompt=prompt)
        result = chain.run(product="colorful socks", callbacks=[callback])

        assert result is not None
        assert mock_send_interaction.call_count == 1
        interaction_watch = mock_send_interaction.call_args[0][0]
        assert isinstance(interaction_watch, InteractionWatch)
        assert interaction_watch.input == "product: colorful socks"
        assert interaction_watch.output == "text: Sample langchain response"
        assert interaction_watch.end_user == "test_user"
        assert len(interaction_watch.spans) == 2
        assert len(interaction_watch.hierarchy) == 2
        for span in interaction_watch.spans:
            assert isinstance(span, SpanWatch)
            assert span.provider_extras is not None
            if span.module == "langchain":
                assert span.provider_extras.get("event_type") is not None
        assert (
            json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder) is not None
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


def test_langchain_chat_chain__callback_on_model(openai_chat: dict[str, Any]) -> None:
    with patch("openai.ChatCompletion.create") as mock_chat_completion_create, patch(
        "nebuly.providers.langchain.post_message",
        return_value=Mock(),
    ) as mock_send_interaction:
        mock_chat_completion_create.return_value = openai_chat
        callback = LangChainTrackingHandler(
            api_key="test_key",
            user_id="test_user",
        )
        llm = ChatOpenAI(openai_api_key="test", callbacks=[callback])

        chat_prompt = ChatPromptTemplate.from_messages(
            messages=[
                ("user", "Hello! I am {name}"),
                ("assistant", "Hi there! How can I assist you today?"),
                ("user", "I need help with my computer."),
            ]
        )
        chain = LLMChain(llm=llm, prompt=chat_prompt)
        result = chain.run(
            prompt=chat_prompt,
            name="Valerio",
        )

        assert result is not None
        assert mock_send_interaction.call_count == 1
        interaction_watch = mock_send_interaction.call_args[0][0]
        assert isinstance(interaction_watch, InteractionWatch)
        assert interaction_watch.input == "I need help with my computer."
        assert interaction_watch.history == [
            HistoryEntry(
                user="Hello! I am Valerio",
                assistant="Hi there! How can I assist you today?",
            ),
        ]
        assert interaction_watch.output == "Hi there! How can I assist you today?"
        assert len(interaction_watch.spans) == 1
        assert len(interaction_watch.hierarchy) == 1
        for span in interaction_watch.spans:
            assert isinstance(span, SpanWatch)
            assert span.provider_extras is not None
            if span.module == "langchain":
                assert span.provider_extras.get("event_type") is not None

        assert (
            json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder) is not None
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
    with patch("openai.ChatCompletion.create") as mock_chat_completion_create, patch(
        "nebuly.providers.langchain.post_message",
        return_value=Mock(),
    ) as mock_send_interaction:
        mock_chat_completion_create.side_effect = [
            openai_chat_with_function,
            openai_chat,
        ]
        callback = LangChainTrackingHandler(
            api_key="test_key",
            user_id="test_user",
        )

        llm = ChatOpenAI(temperature=0, openai_api_key="test")

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
        agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)  # type: ignore
        agent_executor = AgentExecutor(agent=agent, tools=tools)  # type: ignore
        agent_executor.run(
            input="how many letters in the word educa?", callbacks=[callback]
        )

        assert mock_send_interaction.call_count == 1
        interaction_watch = mock_send_interaction.call_args[0][0]
        assert isinstance(interaction_watch, InteractionWatch)
        assert interaction_watch.input == "input: how many letters in the word educa?"
        assert interaction_watch.history == []
        assert (
            interaction_watch.output == "output: Hi there! How can I assist you today?"
        )
        assert len(interaction_watch.spans) == 4
        assert len(interaction_watch.hierarchy) == 4
        for span in interaction_watch.spans:
            assert isinstance(span, SpanWatch)
            if span.module == "langchain":
                assert span.provider_extras is not None
                event_type = span.provider_extras.get("event_type")
                assert event_type is not None
                if event_type == EventType.TOOL.value:
                    assert span.rag_source == "get_word_length"

        assert (
            json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder) is not None
        )


def test_langchain_sequential_chain_single_input_var(
    openai_completion: dict[str, Any]
) -> None:
    with patch("openai.Completion.create") as mock_completion_create, patch(
        "nebuly.providers.langchain.post_message",
        return_value=Mock(),
    ) as mock_send_interaction:
        mock_completion_create.return_value = openai_completion
        callback = LangChainTrackingHandler(
            api_key="test_key",
            user_id="test_user",
        )
        llm = OpenAI(temperature=0.7, openai_api_key="test")
        synopsis_template = """
        Title: {title}
        Playwright: This is a synopsis for the above play:"""
        synopsis_prompt_template = PromptTemplate(
            input_variables=["title"], template=synopsis_template
        )
        synopsis_chain = LLMChain(
            llm=llm, prompt=synopsis_prompt_template, output_key="synopsis"
        )
        llm = OpenAI(temperature=0.7, openai_api_key="test")
        template = """
        Play Synopsis:
        {synopsis}
        Review from a New York Times play critic of the above play:"""
        prompt_template = PromptTemplate(
            input_variables=["synopsis"], template=template
        )
        review_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="review")

        overall_chain = SequentialChain(
            chains=[synopsis_chain, review_chain],
            input_variables=["title"],
            # Here we return multiple variables
            output_variables=["synopsis", "review"],
        )

        overall_chain("Tragedy at sunset on the beach", callbacks=[callback])

        assert mock_send_interaction.call_count == 1
        interaction_watch = mock_send_interaction.call_args[0][0]
        assert isinstance(interaction_watch, InteractionWatch)
        assert "Tragedy at sunset on the beach" in interaction_watch.input
        assert isinstance(interaction_watch.output, str)
        assert "review" in interaction_watch.output


def test_langchain_sequential_chain_multiple_input_vars(
    openai_completion: dict[str, Any]
) -> None:
    with patch("openai.Completion.create") as mock_completion_create, patch(
        "nebuly.providers.langchain.post_message",
        return_value=Mock(),
    ) as mock_send_interaction:
        mock_completion_create.return_value = openai_completion
        callback = LangChainTrackingHandler(
            api_key="test_key",
            user_id="test_user",
        )
        llm = OpenAI(temperature=0.7, openai_api_key="test")
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
        llm = OpenAI(temperature=0.7, openai_api_key="test")
        template = """
        Play Synopsis:
        {synopsis}
        Review from a New York Times play critic of the above play:"""
        prompt_template = PromptTemplate(
            input_variables=["synopsis"], template=template
        )
        review_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="review")

        overall_chain = SequentialChain(
            chains=[synopsis_chain, review_chain],
            input_variables=["era", "title"],
            # Here we return multiple variables
            output_variables=["synopsis", "review"],
        )

        overall_chain(
            {"title": "Tragedy at sunset on the beach", "era": "Victorian England"},
            callbacks=[callback],
        )

        assert mock_send_interaction.call_count == 1
        interaction_watch = mock_send_interaction.call_args[0][0]
        assert isinstance(interaction_watch, InteractionWatch)
        assert "Victorian England" in interaction_watch.input
        assert isinstance(interaction_watch.output, str)
        assert "review" in interaction_watch.output


def test_langchain_llm_chain__lcel__callback_on_model(
    openai_completion: dict[str, Any]
) -> None:
    with patch("openai.Completion.create") as mock_completion_create, patch(
        "nebuly.providers.langchain.post_message",
        return_value=Mock(),
    ) as mock_send_interaction:
        mock_completion_create.return_value = openai_completion
        callback = LangChainTrackingHandler(
            api_key="test_key",
            user_id="test_user",
        )
        prompt = PromptTemplate.from_template(
            "What is a good name for a company that makes {product}?"
        )
        runnable = (
            prompt
            | OpenAI(openai_api_key="test", callbacks=[callback])
            | StrOutputParser()
        )
        result = runnable.invoke({"product": "colorful socks"})

        assert result is not None
        assert mock_send_interaction.call_count == 1
        interaction_watch = mock_send_interaction.call_args[0][0]
        assert isinstance(interaction_watch, InteractionWatch)
        assert (
            interaction_watch.input
            == "What is a good name for a company that makes colorful socks?"
        )
        assert interaction_watch.output == "Sample langchain response"
        assert interaction_watch.end_user == "test_user"
        assert len(interaction_watch.spans) == 1
        assert len(interaction_watch.hierarchy) == 1
        for span in interaction_watch.spans:
            assert isinstance(span, SpanWatch)
            assert span.provider_extras is not None
            if span.module == "langchain":
                assert span.provider_extras.get("event_type") is not None
        assert (
            json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder) is not None
        )


def test_langchain_chat_chain__lcel__callback_on_model(
    openai_chat: dict[str, Any]
) -> None:
    with patch("openai.ChatCompletion.create") as mock_completion_create, patch(
        "nebuly.providers.langchain.post_message",
        return_value=Mock(),
    ) as mock_send_interaction:
        mock_completion_create.return_value = openai_chat
        callback = LangChainTrackingHandler(
            api_key="test_key",
            user_id="test_user",
        )
        prompt = PromptTemplate.from_template(
            "What is a good name for a company that makes {product}?"
        )
        runnable = (
            prompt
            | ChatOpenAI(openai_api_key="test", callbacks=[callback])
            | StrOutputParser()
        )
        result = runnable.invoke({"product": "colorful socks"})

        assert result is not None
        assert mock_send_interaction.call_count == 1
        interaction_watch = mock_send_interaction.call_args[0][0]
        assert isinstance(interaction_watch, InteractionWatch)
        assert (
            interaction_watch.input
            == "What is a good name for a company that makes colorful socks?"
        )
        assert interaction_watch.output == "Hi there! How can I assist you today?"
        assert interaction_watch.end_user == "test_user"
        assert len(interaction_watch.spans) == 1
        assert len(interaction_watch.hierarchy) == 1
        for span in interaction_watch.spans:
            assert isinstance(span, SpanWatch)
            assert span.provider_extras is not None
            if span.module == "langchain":
                assert span.provider_extras.get("event_type") is not None
        assert (
            json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder) is not None
        )
