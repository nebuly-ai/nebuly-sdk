# isort:skip_file
# pylint: disable=ungrouped-imports,wrong-import-order

from __future__ import annotations

import json
import os
from unittest.mock import Mock, patch

import llama_index.core
import pytest
import numpy as np

from nebuly.providers.llama_index import LlamaIndexTrackingHandler

from llama_index.core.indices import load_index_from_storage
from llama_index.core.readers import download_loader
from llama_index.core.storage import (
    StorageContext,
)
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.llms.openai import OpenAI  # type: ignore
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from nebuly.entities import InteractionWatch, SpanWatch, HistoryEntry
from nebuly.requests import CustomJSONEncoder

os.environ["OPENAI_API_KEY"] = "test_key"
llama_index.core.global_handler = LlamaIndexTrackingHandler(
    api_key="nb-748b415156ef9cfa46c2e371fff27a6a0c42d6b4a3f1ff9d",
    user_id="test_user",
    nebuly_tags={"tenant": "ciao"},
    feature_flags=["test"],
)


@pytest.fixture(name="openai_embedding")
def fixture_openai_embedding() -> list[float]:
    return list(np.random.uniform(low=-0.1, high=0.1, size=(1536,)))


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


def test_llm_completion(openai_chat_completion: ChatCompletion) -> None:
    with patch(
        "openai.resources.chat.completions.Completions.create"
    ) as mock_completion_create, patch(
        "nebuly.providers.llama_index.post_message",
        return_value=Mock(),
    ) as mock_send_interaction:
        mock_completion_create.return_value = openai_chat_completion

        result = OpenAI().complete("Paul Graham is ")

        assert result is not None
        assert mock_send_interaction.call_count == 1
        interaction_watch = mock_send_interaction.call_args[0][0]
        assert isinstance(interaction_watch, InteractionWatch)
        assert interaction_watch.input == "Paul Graham is "
        assert interaction_watch.output == "Italian"
        assert interaction_watch.end_user == "test_user"
        assert len(interaction_watch.spans) == 1
        assert len(interaction_watch.hierarchy) == 1
        for span in interaction_watch.spans:
            assert isinstance(span, SpanWatch)
            assert span.provider_extras is not None
            if span.module == "llama_index":
                assert span.provider_extras.get("event_type") is not None
        assert (
            json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder) is not None
        )


def test_llm_chat(openai_chat_completion: ChatCompletion) -> None:
    with patch(
        "openai.resources.chat.completions.Completions.create"
    ) as mock_completion_create, patch(
        "nebuly.providers.llama_index.post_message",
        return_value=Mock(),
    ) as mock_send_interaction:
        mock_completion_create.return_value = openai_chat_completion

        result = OpenAI().chat(
            messages=[
                ChatMessage(role=MessageRole.USER, content="Hello"),
                ChatMessage(role=MessageRole.ASSISTANT, content="Hello, how are you?"),
                ChatMessage(role=MessageRole.USER, content="I am fine, thanks"),
            ]
        )

        assert result is not None
        assert mock_send_interaction.call_count == 1
        interaction_watch = mock_send_interaction.call_args[0][0]
        assert isinstance(interaction_watch, InteractionWatch)
        assert interaction_watch.input == "I am fine, thanks"
        assert interaction_watch.output == "Italian"
        assert interaction_watch.history == [
            HistoryEntry(user="Hello", assistant="Hello, how are you?"),
        ]
        assert interaction_watch.end_user == "test_user"
        assert interaction_watch.tags == {"tenant": "ciao"}
        assert interaction_watch.feature_flags == ["test"]
        assert len(interaction_watch.spans) == 1
        assert len(interaction_watch.hierarchy) == 1
        for span in interaction_watch.spans:
            assert isinstance(span, SpanWatch)
            assert span.provider_extras is not None
            if span.module == "llama_index":
                assert span.provider_extras.get("event_type") is not None
        assert (
            json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder) is not None
        )


def test_query(
    openai_chat_completion: ChatCompletion, openai_embedding: list[float]
) -> None:
    with patch(
        "openai.resources.chat.completions.Completions.create"
    ) as mock_completion_create, patch(
        "nebuly.providers.llama_index.post_message",
        return_value=Mock(),
    ) as mock_send_interaction, patch(
        "openai.resources.embeddings.Embeddings.create"
    ) as mock_embedding_create:
        mock_completion_create.return_value = openai_chat_completion
        mock_embedding_create.return_value = Mock(
            data=[Mock(embedding=openai_embedding)]
        )
        SimpleWebPageReader = download_loader("SimpleWebPageReader")
        assert SimpleWebPageReader is not None
        storage_context = StorageContext.from_defaults(
            persist_dir="tests/providers/test_index"
        )
        index = load_index_from_storage(storage_context)
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


def test_chat(
    openai_chat_completion: ChatCompletion, openai_embedding: list[float]
) -> None:
    with patch(
        "openai.resources.chat.completions.Completions.create"
    ) as mock_completion_create, patch(
        "nebuly.providers.llama_index.post_message",
        return_value=Mock(),
    ) as mock_send_interaction, patch(
        "openai.resources.embeddings.Embeddings.create"
    ) as mock_embedding_create:
        mock_completion_create.return_value = openai_chat_completion
        mock_embedding_create.return_value = Mock(
            data=[Mock(embedding=openai_embedding)]
        )
        mock_completion_create.return_value = openai_chat_completion
        SimpleWebPageReader = download_loader("SimpleWebPageReader")
        assert SimpleWebPageReader is not None
        storage_context = StorageContext.from_defaults(
            persist_dir="tests/providers/test_index"
        )
        index = load_index_from_storage(storage_context)
        chat_engine = index.as_chat_engine()
        result = chat_engine.chat(
            "What language is on this website?",
            chat_history=[
                ChatMessage(role=MessageRole.USER, content="Hello"),
                ChatMessage(role=MessageRole.ASSISTANT, content="Hello, how are you?"),
            ],
        )

        assert result is not None
        assert mock_send_interaction.call_count == 1
        interaction_watch = mock_send_interaction.call_args[0][0]
        assert isinstance(interaction_watch, InteractionWatch)
        assert interaction_watch.input == "What language is on this website?"
        assert interaction_watch.output == "Italian"
        assert interaction_watch.end_user == "test_user"
        assert interaction_watch.history == [
            HistoryEntry(user="Hello", assistant="Hello, how are you?"),
        ]
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
