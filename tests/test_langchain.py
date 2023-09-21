from unittest.mock import patch

import pytest
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

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
            for span in interaction_watch.spans:
                assert isinstance(span, SpanWatch)
