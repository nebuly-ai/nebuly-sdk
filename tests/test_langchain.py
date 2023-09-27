from unittest.mock import patch

import langchain
import pytest
from langchain.agents import AgentExecutor, OpenAIFunctionsAgent, tool
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import SystemMessage

from nebuly.contextmanager import new_interaction
from nebuly.entities import EventType, InteractionWatch, SpanWatch
from nebuly.observers import NebulyObserver
from tests import common

# Cache original functions
orig_func_llm_gen = langchain.llms.base.BaseLLM.generate
orig_func_chat_gen = langchain.chat_models.base.BaseChatModel.generate


def nebuly_init(observer):
    # Reset original functions
    langchain.llms.base.BaseLLM.generate = orig_func_llm_gen
    langchain.chat_models.base.BaseChatModel.generate = orig_func_chat_gen
    common.nebuly_init(observer)


@pytest.fixture()
def openai_completion() -> dict:
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


def test_langchain_llm_chain__no_context_manager(openai_completion: dict) -> None:
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
            result = chain.run("colorful socks")

            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "colorful socks"
            assert interaction_watch.output == {
                "product": "colorful socks",
                "text": "Sample langchain response",
            }
            assert len(interaction_watch.spans) == 3
            assert len(interaction_watch.hierarchy) == 3
            for span in interaction_watch.spans:
                assert isinstance(span, SpanWatch)
                if span.module == "langchain":
                    assert span.provider_extras.get("event_type") is not None


def test_langchain_llm_chain__with_context_manager(openai_completion: dict) -> None:
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
            with new_interaction(user="test", group_profile="tier 1") as interaction:
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
                if span.module == "langchain":
                    assert span.provider_extras.get("event_type") is not None


def test_langchain_llm_chain__multiple_chains_in_interaction(
    openai_completion: dict,
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
            with new_interaction(user="test", group_profile="tier 1") as interaction:
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
                if span.module == "langchain":
                    assert span.provider_extras.get("event_type") is not None


def test_langchain_llm_chain__multiple_interactions(openai_completion: dict) -> None:
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
            with new_interaction(user="test", group_profile="tier 1") as interaction:
                interaction.set_input("colorful socks")
                result_0 = chain.run("colorful socks")
                interaction.set_output("Sample langchain response 1")
            with new_interaction(user="test", group_profile="tier 1") as interaction:
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
                if span.module == "langchain":
                    assert span.provider_extras.get("event_type") is not None


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


def test_langchain_chat_chain__no_context_manager(openai_chat: dict) -> None:
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
            assert interaction_watch.output == {
                "name": "Valerio",
                "prompt": chat_prompt,
                "text": "Hi there! How can I assist you today?",
            }
            assert len(interaction_watch.spans) == 3
            assert len(interaction_watch.hierarchy) == 3
            for span in interaction_watch.spans:
                assert isinstance(span, SpanWatch)
                if span.module == "langchain":
                    assert span.provider_extras.get("event_type") is not None


@pytest.fixture()
def openai_chat_with_function() -> dict:
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
    openai_chat_with_function: dict, openai_chat: dict
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
            agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
            agent_executor = AgentExecutor(agent=agent, tools=tools)
            agent_executor.run("how many letters in the word educa?")

            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == "how many letters in the word educa?"
            assert interaction_watch.history is None
            assert interaction_watch.output == {
                "input": "how many letters in the word educa?",
                "output": "Hi there! How can I assist you today?",
            }
            assert len(interaction_watch.spans) == 6
            assert len(interaction_watch.hierarchy) == 6
            for span in interaction_watch.spans:
                assert isinstance(span, SpanWatch)
                if span.module == "langchain":
                    event_type = span.provider_extras.get("event_type")
                    assert event_type is not None
                    if event_type == EventType.TOOL.value:
                        assert span.rag_source == "get_word_length"
