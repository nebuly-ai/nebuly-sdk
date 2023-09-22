from unittest.mock import patch

import pytest
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate

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
            nebuly.init(api_key="test")
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


def test_langchain_llm_chain__with_context_manager(openai_completion: dict) -> None:
    with patch("openai.Completion.create") as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = openai_completion
            nebuly.init(api_key="test")
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


def test_langchain_llm_chain__multiple_chains_in_interaction(
    openai_completion: dict,
) -> None:
    with patch("openai.Completion.create") as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = openai_completion
            nebuly.init(api_key="test")
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


def test_langchain_llm_chain__multiple_interactions(openai_completion: dict) -> None:
    with patch("openai.Completion.create") as mock_completion_create:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion_create.return_value = openai_completion
            nebuly.init(api_key="test")
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
            nebuly.init(api_key="test")
            llm = ChatOpenAI()

            chat_prompt = ChatPromptTemplate.from_messages(
                messages=[
                    ("user", "Hello!"),
                    ("assistant", "Hi there! How can I assist you today?"),
                    ("user", "I need help with my computer."),
                ]
            )
            chain = LLMChain(llm=llm, prompt=chat_prompt)
            result = chain.run(
                prompt=chat_prompt,
            )

            assert result is not None
            assert mock_observer.call_count == 1
            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.input == [
                ("user", "Hello!"),
                ("assistant", "Hi there! How can I assist you today?"),
                ("user", "I need help with my computer."),
            ]
            assert interaction_watch.output == "Hi there! How can I assist you today?"
            assert len(interaction_watch.spans) == 3
            assert len(interaction_watch.hierarchy) == 3
            for span in interaction_watch.spans:
                assert isinstance(span, SpanWatch)
