# isort:skip_file
# pylint: disable=ungrouped-imports,wrong-import-order
import json
from unittest.mock import Mock, patch

from nebuly.providers.llama_index import NebulyTrackingHandler

import llama_index
import pytest
from llama_index import VectorStoreIndex, download_loader
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from nebuly.entities import InteractionWatch, SpanWatch
from nebuly.requests import CustomJSONEncoder

llama_index.global_handler = NebulyTrackingHandler(
    api_key="nb-748b415156ef9cfa46c2e371fff27a6a0c42d6b4a3f1ff9d", user_id="test_user"
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
                    content="Italian", role="assistant", function_call=None
                ),
            )
        ],
        created=1696506982,
        model="gpt-3.5-turbo-0613",
        object="chat.completion",
        usage=CompletionUsage(completion_tokens=5, prompt_tokens=12, total_tokens=17),
    )


def test_query(openai_chat_completion: ChatCompletion) -> None:
    with patch(
        "openai.resources.chat.completions.Completions.create"
    ) as mock_completion_create, patch(
        "nebuly.providers.llama_index.post_message",
        return_value=Mock(),
    ) as mock_send_interaction:
        mock_completion_create.return_value = openai_chat_completion
        SimpleWebPageReader = download_loader("SimpleWebPageReader")

        loader = SimpleWebPageReader()
        documents = loader.load_data(urls=["https://google.com"])
        index = VectorStoreIndex.from_documents(documents)

        query_engine = index.as_query_engine()
        result = query_engine.query("What language is on this website?")

        assert result is not None
        assert mock_send_interaction.call_count == 1
        interaction_watch = mock_send_interaction.call_args[0][0]
        assert isinstance(interaction_watch, InteractionWatch)
        assert interaction_watch.input == "What language is on this website?"
        assert interaction_watch.output == "Italian"
        assert interaction_watch.end_user == "test_user"
        assert len(interaction_watch.spans) == 6
        assert len(interaction_watch.hierarchy) == 6
        rag_sources = []
        for span in interaction_watch.spans:
            assert isinstance(span, SpanWatch)
            assert span.provider_extras is not None
            if span.module == "llama_index":
                assert span.provider_extras.get("event_type") is not None
            if span.rag_source is not None:
                rag_sources.append(span.rag_source)
        assert "SimpleWebPageReader" in rag_sources
        assert (
            json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder) is not None
        )


def test_chat(openai_chat_completion: ChatCompletion) -> None:
    with patch(
        "openai.resources.chat.completions.Completions.create"
    ) as mock_completion_create, patch(
        "nebuly.providers.llama_index.post_message",
        return_value=Mock(),
    ) as mock_send_interaction:
        mock_completion_create.return_value = openai_chat_completion
        SimpleWebPageReader = download_loader("SimpleWebPageReader")

        loader = SimpleWebPageReader()
        documents = loader.load_data(urls=["https://google.com"])
        index = VectorStoreIndex.from_documents(documents)

        chat_engine = index.as_chat_engine()
        result = chat_engine.chat("Tell me a language")

        assert result is not None
        assert mock_send_interaction.call_count == 1
        interaction_watch = mock_send_interaction.call_args[0][0]
        assert isinstance(interaction_watch, InteractionWatch)
        assert interaction_watch.input == "Tell me a language"
        assert interaction_watch.output == "Italian"
        assert interaction_watch.end_user == "test_user"
        assert len(interaction_watch.spans) == 2
        assert len(interaction_watch.hierarchy) == 2
        for span in interaction_watch.spans:
            assert isinstance(span, SpanWatch)
            assert span.provider_extras is not None
            if span.module == "llama_index":
                assert span.provider_extras.get("event_type") is not None
        assert (
            json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder) is not None
        )
