import uuid
from typing import Any

import pytest
from langchain.schema import Document
from langchain.schema.messages import AIMessage, HumanMessage, SystemMessage
from langchain.schema.output import ChatGeneration, Generation, LLMResult

from nebuly.event_pairing_dispatchers import (
    EventData,
    EventPairingDispatcher,
    EventType,
)


def test_new_event_pairing_dispatcher() -> None:
    event_dispatcher = EventPairingDispatcher()
    assert event_dispatcher is not None


@pytest.fixture(name="simple_sequential_chain_serialized_data")
def fixture_simple_sequential_chain_serialized_data() -> dict[str, Any]:
    return {
        "lc": 1,
        "type": "not_implemented",
        "id": ["langchain", "chains", "sequential", "SimpleSequentialChain"],
        "repr": "SimpleSequentialChain(chains=[synopsis_chain, review_chain], ..."
        "verbose=True)",
    }


@pytest.fixture(name="llm_chain_serialized_data")
def fixture_llm_chain_serialized_data() -> dict[str, Any]:
    return {
        "lc": 1,
        "type": "not_implemented",
        "id": ["langchain", "chains", "llm", "LLMChain"],
        "kwargs": {
            "llm": {
                "lc": 1,
                "type": "constructor",
                "id": ["langchain", "llms", "openai", "OpenAI"],
                "kwargs": {
                    "temperature": 0.7,
                    "openai_api_key": {
                        "lc": 1,
                        "type": "secret",
                        "id": ["OPENAI_API_KEY"],
                    },
                },
            },
            "prompt": {
                "lc": 1,
                "type": "constructor",
                "id": ["langchain", "prompts", "prompt", "PromptTemplate"],
                "kwargs": {
                    "input_variables": ["title"],
                    "template": "You are a playwright. Given the title of play, it is "
                    "your job to write a synopsis for that title.\n\nTitle:"
                    " {title}\nPlaywright: This is a synopsis for the above"
                    " play:",
                    "template_format": "f-string",
                },
            },
        },
    }


@pytest.fixture(name="agent_chain_serialized_data")
def fixture_agent_chain_serialized_data() -> dict[str, Any]:
    return {
        "lc": 1,
        "type": "not_implemented",
        "id": ["langchain", "agents", "agent", "AgentExecutor"],
        "repr": "AgentExecutor(memory=None, callbacks=None, callback_manager=None ...",
    }


@pytest.fixture(name="qa_chain_serialized_data")
def fixture_qa_chain_serialized_data() -> dict[str, Any]:
    return {
        "lc": 1,
        "type": "not_implemented",
        "id": ["langchain", "chains", "retrieval_qa", "base", "RetrievalQA"],
        "repr": "RetrievalQA(memory=None, callbacks=None, callback_manager=None, ...",
    }


@pytest.fixture(name="tool_serialized_data")
def fixture_tool_serialized_data() -> dict[str, Any]:
    return {
        "description": "useful for when you need to answer questions about "
        "current events",
        "name": "Search",
    }


@pytest.fixture(name="retrieval_serialized_data")
def fixture_retrieval_serialized_data() -> dict[str, Any]:
    return {
        "lc": 1,
        "type": "not_implemented",
        "id": ["langchain", "vectorstores", "base", "VectorStoreRetriever"],
        "repr": "VectorStoreRetriever(tags=['Chroma', 'OpenAIEmbeddings'], "
        "metadata=None, vectorstore=<langchain.vectorstores.chroma.Chroma "
        "object at 0x14db9e790>, search_type='similarity', "
        "search_kwargs={})",
    }


@pytest.fixture(name="llm_serialized_data")
def fixture_llm_serialized_data() -> dict[str, Any]:
    return {
        "id": ["langchain", "llms", "openai", "OpenAI"],
        "kwargs": {
            "openai_api_key": {"id": ["OPENAI_API_KEY"], "lc": 1, "type": "secret"},
            "temperature": 0.7,
        },
        "lc": 1,
        "type": "constructor",
    }


@pytest.fixture(name="tool_sample_output")
def fixture_tool_sample_output() -> str:
    return """
        Page: Leonardo DiCaprio
        Summary: Leonardo Wilhelm DiCaprio (, ; Italian: [diˈkaːprjo]; born
        November 11, 1974) is an American actor ...

        Page: Camila Morrone
        Summary: Camila Rebeca Morrone Polak (born June 16, 1997) is an American
        actress and model...

        Page: The Departed
        Summary: The Departed is a 2006 American crime thriller film directed by
        Martin Scorsese...
        """


@pytest.fixture(name="default_kwargs")
def fixture_default_kwargs() -> dict[str, Any]:
    return {"metadata": {}, "name": None, "tags": []}


@pytest.fixture(name="default_tool_kwargs")
def fixture_default_tool_kwargs() -> dict[str, Any]:
    return {
        "color": "green",
        "llm_prefix": "Thought:",
        "metadata": {},
        "observation_prefix": "Observation: ",
        "tags": [],
    }


def test_event_pairing_dispatcher_on_chain_start__new_root_chain(
    simple_sequential_chain_serialized_data: dict[str, Any],
    default_kwargs: dict[str, Any],
) -> None:
    run_id = uuid.uuid4()
    event_dispatcher = EventPairingDispatcher()
    event_dispatcher.on_chain_start(
        serialized=simple_sequential_chain_serialized_data,
        inputs={"input": "Tragedy at sunset on the beach"},
        run_id=run_id,
        parent_run_id=None,
        **default_kwargs
    )

    assert event_dispatcher.events_storage.events is not None
    assert len(event_dispatcher.events_storage.events) == 1
    assert run_id in event_dispatcher.events_storage.events
    assert event_dispatcher.events_storage.events[run_id].event_id == run_id
    assert event_dispatcher.events_storage.events[run_id].hierarchy is None
    assert event_dispatcher.events_storage.events[run_id].data == EventData(
        type=EventType.CHAIN,
        serialized=simple_sequential_chain_serialized_data,
        inputs={"input": "Tragedy at sunset on the beach"},
        input_extras=default_kwargs,
    )


@pytest.fixture(name="llm_chain_kwargs")
def fixture_llm_chain_kwargs() -> dict[str, Any]:
    return {"metadata": {}, "name": None, "tags": ["step_1"]}


def test_event_pairing_dispatcher_on_chain_start__existing_root_chain(
    simple_sequential_chain_serialized_data: dict[str, Any],
    llm_chain_serialized_data: dict[str, Any],
    default_kwargs: dict[str, Any],
    llm_chain_kwargs: dict[str, Any],
) -> None:
    root_id = uuid.uuid4()
    event_dispatcher = EventPairingDispatcher()
    event_dispatcher.on_chain_start(
        serialized=simple_sequential_chain_serialized_data,
        inputs={"input": "Tragedy at sunset on the beach"},
        run_id=root_id,
        parent_run_id=None,
        **default_kwargs
    )

    run_id = uuid.uuid4()
    event_dispatcher.on_chain_start(
        serialized=llm_chain_serialized_data,
        inputs={"title": "Tragedy at sunset on the beach"},
        run_id=run_id,
        parent_run_id=root_id,
        **llm_chain_kwargs
    )

    assert event_dispatcher.events_storage.events is not None
    assert len(event_dispatcher.events_storage.events) == 2
    assert run_id in event_dispatcher.events_storage.events
    assert event_dispatcher.events_storage.events[run_id].event_id == run_id
    assert event_dispatcher.events_storage.events[run_id].hierarchy is not None
    assert event_dispatcher.events_storage.events[run_id].hierarchy.parent_run_id == root_id  # type: ignore # pylint: disable=line-too-long
    assert event_dispatcher.events_storage.events[run_id].hierarchy.root_run_id == root_id  # type: ignore # pylint: disable=line-too-long
    assert event_dispatcher.events_storage.events[run_id].data == EventData(
        type=EventType.CHAIN,
        serialized=llm_chain_serialized_data,
        inputs={"title": "Tragedy at sunset on the beach"},
        input_extras=llm_chain_kwargs,
    )


def test_event_pairing_dispatcher_on_chain_end__root_chain(
    simple_sequential_chain_serialized_data: dict[str, Any],
    default_kwargs: dict[str, Any],
) -> None:
    run_id = uuid.uuid4()
    event_dispatcher = EventPairingDispatcher()
    event_dispatcher.on_chain_start(
        serialized=simple_sequential_chain_serialized_data,
        inputs={"input": "Tragedy at sunset on the beach"},
        run_id=run_id,
        parent_run_id=None,
        **default_kwargs
    )

    output_kwargs = {"tags": ["step_1"]}
    watched_event = event_dispatcher.on_chain_end(
        outputs={"text": "Tragedy at Sunset on the Beach"},
        run_id=run_id,
        parent_run_id=None,
        **output_kwargs
    )

    assert event_dispatcher.events_storage.events is not None
    assert len(event_dispatcher.events_storage.events) == 0

    assert watched_event is not None
    assert watched_event.run_id == run_id
    assert watched_event.hierarchy is None
    assert watched_event.inputs == {"input": "Tragedy at sunset on the beach"}
    assert watched_event.outputs == {"text": "Tragedy at Sunset on the Beach"}
    assert watched_event.type is EventType.CHAIN
    assert watched_event.serialized == simple_sequential_chain_serialized_data
    assert watched_event.extras is not None
    assert watched_event.extras.input == default_kwargs
    assert watched_event.extras.output == output_kwargs


def test_event_pairing_dispatcher_on_chain_end__child_chain(
    simple_sequential_chain_serialized_data: dict[str, Any],
    llm_chain_serialized_data: dict[str, Any],
    default_kwargs: dict[str, Any],
) -> None:
    root_id = uuid.uuid4()
    event_dispatcher = EventPairingDispatcher()
    event_dispatcher.on_chain_start(
        serialized=simple_sequential_chain_serialized_data,
        inputs={"input": "Tragedy at sunset on the beach"},
        run_id=root_id,
        parent_run_id=None,
        **default_kwargs
    )

    run_id = uuid.uuid4()
    event_dispatcher.on_chain_start(
        serialized=llm_chain_serialized_data,
        inputs={"title": "Tragedy at sunset on the beach"},
        run_id=run_id,
        parent_run_id=root_id,
        **default_kwargs
    )

    output_kwargs = {"tags": ["step_1"]}
    watched_event = event_dispatcher.on_chain_end(
        outputs={"text": "Tragedy at Sunset on the Beach"},
        run_id=run_id,
        parent_run_id=root_id,
        **output_kwargs
    )

    assert event_dispatcher.events_storage.events is not None
    assert len(event_dispatcher.events_storage.events) == 2
    assert root_id in event_dispatcher.events_storage.events
    assert run_id in event_dispatcher.events_storage.events

    assert watched_event is not None
    assert watched_event.run_id == run_id
    assert watched_event.hierarchy is not None
    assert watched_event.hierarchy.parent_run_id == root_id
    assert watched_event.hierarchy.root_run_id == root_id
    assert watched_event.inputs == {"title": "Tragedy at sunset on the beach"}
    assert watched_event.outputs == {"text": "Tragedy at Sunset on the Beach"}
    assert watched_event.type is EventType.CHAIN
    assert watched_event.serialized == llm_chain_serialized_data
    assert watched_event.extras is not None
    assert watched_event.extras.input == default_kwargs
    assert watched_event.extras.output == output_kwargs


def test_event_pairing_dispatcher_on_tool_start(
    agent_chain_serialized_data: dict[str, Any],
    tool_serialized_data: dict[str, Any],
    default_kwargs: dict[str, Any],
    default_tool_kwargs: dict[str, Any],
) -> None:
    chain_id = uuid.uuid4()
    event_dispatcher = EventPairingDispatcher()
    event_dispatcher.on_chain_start(
        serialized=agent_chain_serialized_data,
        inputs={"input": "Who is Leo DiCaprio's girlfriend?"},
        run_id=chain_id,
        parent_run_id=None,
        **default_kwargs
    )

    parent_tool_id = uuid.uuid4()

    event_dispatcher.on_tool_start(
        serialized={
            "description": "useful for when you need to answer questions about current "
            "events",
            "name": "Search",
        },
        input_str="Leonardo DiCaprio's girlfriend",
        run_id=parent_tool_id,
        parent_run_id=chain_id,
        **default_tool_kwargs
    )

    run_id = uuid.uuid4()
    event_dispatcher.on_tool_start(
        serialized=tool_serialized_data,
        input_str="Leonardo DiCaprio's girlfriend",
        run_id=run_id,
        parent_run_id=parent_tool_id,
        **default_tool_kwargs
    )

    assert event_dispatcher.events_storage.events is not None
    assert len(event_dispatcher.events_storage.events) == 3
    assert run_id in event_dispatcher.events_storage.events
    assert (
        event_dispatcher.events_storage.events[parent_tool_id].event_id
        == parent_tool_id
    )
    assert event_dispatcher.events_storage.events[parent_tool_id].hierarchy is not None
    assert event_dispatcher.events_storage.events[parent_tool_id].hierarchy.parent_run_id == chain_id  # type: ignore # pylint: disable=line-too-long
    assert event_dispatcher.events_storage.events[parent_tool_id].data == EventData(
        type=EventType.TOOL,
        serialized=tool_serialized_data,
        inputs={"query": "Leonardo DiCaprio's girlfriend"},
        input_extras=default_tool_kwargs,
    )
    assert event_dispatcher.events_storage.events[run_id].event_id == run_id
    assert event_dispatcher.events_storage.events[run_id].hierarchy is not None
    assert event_dispatcher.events_storage.events[run_id].hierarchy.parent_run_id == parent_tool_id  # type: ignore # pylint: disable=line-too-long
    assert event_dispatcher.events_storage.events[run_id].data == EventData(
        type=EventType.TOOL,
        serialized=tool_serialized_data,
        inputs={"query": "Leonardo DiCaprio's girlfriend"},
        input_extras=default_tool_kwargs,
    )


def test_event_pairing_dispatcher_on_tool_end(
    agent_chain_serialized_data: dict[str, Any],
    tool_serialized_data: dict[str, Any],
    tool_sample_output: str,
    default_kwargs: dict[str, Any],
    default_tool_kwargs: dict[str, Any],
) -> None:
    chain_id = uuid.uuid4()
    event_dispatcher = EventPairingDispatcher()
    event_dispatcher.on_chain_start(
        serialized=agent_chain_serialized_data,
        inputs={"input": "Who is Leo DiCaprio's girlfriend?"},
        run_id=chain_id,
        parent_run_id=None,
        **default_kwargs
    )

    parent_tool_id = uuid.uuid4()

    event_dispatcher.on_tool_start(
        serialized=tool_serialized_data,
        input_str="Leonardo DiCaprio's girlfriend",
        run_id=parent_tool_id,
        parent_run_id=chain_id,
        **default_tool_kwargs
    )

    run_id = uuid.uuid4()
    event_dispatcher.on_tool_start(
        serialized=tool_serialized_data,
        input_str="Leonardo DiCaprio's girlfriend",
        run_id=run_id,
        parent_run_id=parent_tool_id,
        **default_tool_kwargs
    )

    tool_end_sample_kwargs = {"color": "green", "name": "Wikipedia", "tags": []}

    wiki_watched_event = event_dispatcher.on_tool_end(
        output=tool_sample_output,
        run_id=run_id,
        parent_run_id=parent_tool_id,
        **tool_end_sample_kwargs
    )

    search_watched_event = event_dispatcher.on_tool_end(
        output=tool_sample_output,
        run_id=parent_tool_id,
        parent_run_id=chain_id,
        **tool_end_sample_kwargs
    )

    assert event_dispatcher.events_storage.events is not None
    assert len(event_dispatcher.events_storage.events) == 3
    assert wiki_watched_event is not None
    assert wiki_watched_event.run_id == run_id
    assert wiki_watched_event.hierarchy is not None
    assert wiki_watched_event.hierarchy.parent_run_id == parent_tool_id
    assert wiki_watched_event.hierarchy.root_run_id == chain_id
    assert wiki_watched_event.inputs == {"query": "Leonardo DiCaprio's girlfriend"}
    assert wiki_watched_event.outputs == {"result": tool_sample_output}
    assert wiki_watched_event.type is EventType.TOOL
    assert wiki_watched_event.serialized == tool_serialized_data
    assert wiki_watched_event.extras is not None
    assert wiki_watched_event.extras.input == default_tool_kwargs
    assert wiki_watched_event.extras.output == tool_end_sample_kwargs
    assert search_watched_event is not None
    assert search_watched_event.run_id == parent_tool_id
    assert search_watched_event.hierarchy is not None
    assert search_watched_event.hierarchy.parent_run_id == chain_id
    assert search_watched_event.hierarchy.root_run_id == chain_id
    assert search_watched_event.inputs == {"query": "Leonardo DiCaprio's girlfriend"}
    assert search_watched_event.outputs == {"result": tool_sample_output}
    assert search_watched_event.type is EventType.TOOL
    assert search_watched_event.serialized == tool_serialized_data
    assert search_watched_event.extras is not None
    assert search_watched_event.extras.input == default_tool_kwargs
    assert search_watched_event.extras.output == tool_end_sample_kwargs


def test_event_pairing_dispatcher_on_retriever_start(
    qa_chain_serialized_data: dict[str, Any],
    retrieval_serialized_data: dict[str, Any],
    default_kwargs: dict[str, Any],
) -> None:
    chain_id = uuid.uuid4()
    event_dispatcher = EventPairingDispatcher()
    event_dispatcher.on_chain_start(
        serialized=qa_chain_serialized_data,
        inputs={"query": "What did the president say about Ketanji Brown Jackson"},
        run_id=chain_id,
        parent_run_id=None,
        **default_kwargs
    )

    retriever_id = uuid.uuid4()
    retriever_kwargs = {"metadata": {}, "tags": ["Chroma", "OpenAIEmbeddings"]}

    event_dispatcher.on_retriever_start(
        serialized=retrieval_serialized_data,
        query="What did the president say about Ketanji Brown Jackson",
        run_id=retriever_id,
        parent_run_id=chain_id,
        **retriever_kwargs
    )

    assert event_dispatcher.events_storage.events is not None
    assert len(event_dispatcher.events_storage.events) == 2
    assert retriever_id in event_dispatcher.events_storage.events
    assert event_dispatcher.events_storage.events[retriever_id].event_id == retriever_id
    assert event_dispatcher.events_storage.events[retriever_id].hierarchy is not None
    assert event_dispatcher.events_storage.events[retriever_id].hierarchy.parent_run_id == chain_id  # type: ignore # pylint: disable=line-too-long
    assert event_dispatcher.events_storage.events[retriever_id].data == EventData(
        type=EventType.RETRIEVAL,
        serialized=retrieval_serialized_data,
        inputs={"query": "What did the president say about Ketanji Brown Jackson"},
        input_extras=retriever_kwargs,
    )


def test_event_pairing_dispatcher_on_retriever_end(
    qa_chain_serialized_data: dict[str, Any],
    retrieval_serialized_data: dict[str, Any],
    default_kwargs: dict[str, Any],
) -> None:
    chain_id = uuid.uuid4()
    event_dispatcher = EventPairingDispatcher()
    event_dispatcher.on_chain_start(
        serialized=qa_chain_serialized_data,
        inputs={"query": "What did the president say about Ketanji Brown Jackson"},
        run_id=chain_id,
        parent_run_id=None,
        **default_kwargs
    )

    retriever_id = uuid.uuid4()
    retriever_kwargs = {"metadata": {}, "tags": ["Chroma", "OpenAIEmbeddings"]}

    event_dispatcher.on_retriever_start(
        serialized={
            "lc": 1,
            "type": "not_implemented",
            "id": ["langchain", "vectorstores", "base", "VectorStoreRetriever"],
            "repr": "VectorStoreRetriever(tags=['Chroma', 'OpenAIEmbeddings'], "
            "metadata=None, vectorstore=<langchain.vectorstores.chroma.Chroma "
            "object at 0x14db9e790>, search_type='similarity', "
            "search_kwargs={})",
        },
        query="What did the president say about Ketanji Brown Jackson",
        run_id=retriever_id,
        parent_run_id=chain_id,
        **retriever_kwargs
    )

    retriever_output_documents_sample = [
        Document(
            page_content="Tonight. I call on the Senate to...",
            metadata={"source": "./state_of_the_union.txt"},
        ),
        Document(
            page_content="A former top litigator in private practice...",
            metadata={"source": "./state_of_the_union.txt"},
        ),
        Document(
            page_content="And for our LGBTQ+ Americans, let’s finally...",
            metadata={"source": "./state_of_the_union.txt"},
        ),
        Document(
            page_content="Tonight, I’m announcing a crackdown on ",
            metadata={"source": "./state_of_the_union.txt"},
        ),
    ]

    retrieval_output_kwargs = {"tags": ["Chroma", "OpenAIEmbeddings"]}

    watched_event = event_dispatcher.on_retriever_end(
        documents=retriever_output_documents_sample,
        run_id=retriever_id,
        parent_run_id=chain_id,
        **retrieval_output_kwargs
    )

    assert event_dispatcher.events_storage.events is not None
    assert len(event_dispatcher.events_storage.events) == 2
    assert watched_event is not None
    assert watched_event.run_id == retriever_id
    assert watched_event.hierarchy is not None
    assert watched_event.hierarchy.parent_run_id == chain_id
    assert watched_event.hierarchy.root_run_id == chain_id
    assert watched_event.inputs == {
        "query": "What did the president say about Ketanji Brown Jackson"
    }
    assert watched_event.outputs == {
        "documents": retriever_output_documents_sample,
    }
    assert watched_event.type is EventType.RETRIEVAL
    assert watched_event.serialized == retrieval_serialized_data
    assert watched_event.extras is not None
    assert watched_event.extras.input == retriever_kwargs
    assert watched_event.extras.output == retrieval_output_kwargs


@pytest.fixture(name="llm_invocation_params")
def fixture_llm_invocation_params() -> dict[str, Any]:
    return {
        "_type": "openai",
        "frequency_penalty": 0,
        "logit_bias": {},
        "max_tokens": 256,
        "model_name": "text-davinci-003",
        "n": 1,
        "presence_penalty": 0,
        "request_timeout": None,
        "stop": None,
        "temperature": 0.7,
        "top_p": 1,
    }


def test_event_pairing_dispatcher_on_llm_start(  # pylint: disable=too-many-arguments
    simple_sequential_chain_serialized_data: dict[str, Any],
    llm_chain_serialized_data: dict[str, Any],
    llm_serialized_data: dict[str, Any],
    default_kwargs: dict[str, Any],
    llm_chain_kwargs: dict[str, Any],
    llm_invocation_params: dict[str, Any],
) -> None:
    chain_id = uuid.uuid4()
    event_dispatcher = EventPairingDispatcher()
    event_dispatcher.on_chain_start(
        serialized=simple_sequential_chain_serialized_data,
        inputs={"era": "Victorian England", "title": "Tragedy at sunset on the beach"},
        run_id=chain_id,
        parent_run_id=None,
        **default_kwargs
    )

    llm_chain_id = uuid.uuid4()
    event_dispatcher.on_chain_start(
        serialized=llm_chain_serialized_data,
        inputs={"era": "Victorian England", "title": "Tragedy at sunset on the beach"},
        run_id=llm_chain_id,
        parent_run_id=chain_id,
        **llm_chain_kwargs
    )

    llm_model_id = uuid.uuid4()
    sample_llm_prompts = [
        "You are a playwright. Given the title of play and the era it is set in, "
        "it is your job to write a synopsis for that title.\n\nTitle: Tragedy at "
        "sunset on the beach\nEra: Victorian England\nPlaywright: This is a synopsis "
        "for the above play:"
    ]
    event_dispatcher.on_llm_start(
        serialized=llm_serialized_data,
        prompts=sample_llm_prompts,
        run_id=llm_model_id,
        parent_run_id=llm_chain_id,
        invocation_params=llm_invocation_params,
        **default_kwargs
    )

    assert event_dispatcher.events_storage.events is not None
    assert len(event_dispatcher.events_storage.events) == 3
    assert llm_model_id in event_dispatcher.events_storage.events
    assert event_dispatcher.events_storage.events[llm_model_id].event_id == llm_model_id
    assert event_dispatcher.events_storage.events[llm_model_id].hierarchy is not None
    assert event_dispatcher.events_storage.events[llm_model_id].hierarchy.parent_run_id == llm_chain_id  # type: ignore # pylint: disable=line-too-long
    assert event_dispatcher.events_storage.events[llm_model_id].data == EventData(
        type=EventType.LLM_MODEL,
        serialized=llm_serialized_data,
        inputs={"prompts": sample_llm_prompts},
        input_extras=dict(
            {"invocation_params": llm_invocation_params}, **default_kwargs
        ),
    )


@pytest.fixture(name="chat_serialized_data")
def fixture_chat_serialized_data() -> dict[str, Any]:
    return {
        "id": ["langchain", "chat_models", "openai", "ChatOpenAI"],
        "kwargs": {
            "openai_api_key": {"id": ["OPENAI_API_KEY"], "lc": 1, "type": "secret"}
        },
        "lc": 1,
        "type": "constructor",
    }


@pytest.fixture(name="chat_invocation_params")
def fixture_chat_invocation_params() -> dict[str, Any]:
    return {
        "_type": "openai-chat",
        "max_tokens": None,
        "model": "gpt-3.5-turbo",
        "model_name": "gpt-3.5-turbo",
        "n": 1,
        "request_timeout": None,
        "stop": None,
        "stream": False,
        "temperature": 0.7,
    }


def test_event_pairing_dispatcher_on_chat_model_start(  # pylint: disable=too-many-arguments
    simple_sequential_chain_serialized_data: dict[str, Any],
    llm_chain_serialized_data: dict[str, Any],
    default_kwargs: dict[str, Any],
    llm_chain_kwargs: dict[str, Any],
    chat_serialized_data: dict[str, Any],
    chat_invocation_params: dict[str, Any],
) -> None:
    chain_id = uuid.uuid4()
    event_dispatcher = EventPairingDispatcher()
    event_dispatcher.on_chain_start(
        serialized=simple_sequential_chain_serialized_data,
        inputs={"era": "Victorian England", "title": "Tragedy at sunset on the beach"},
        run_id=chain_id,
        parent_run_id=None,
        **default_kwargs
    )

    llm_chain_id = uuid.uuid4()
    event_dispatcher.on_chain_start(
        serialized=llm_chain_serialized_data,
        inputs={"era": "Victorian England", "title": "Tragedy at sunset on the beach"},
        run_id=llm_chain_id,
        parent_run_id=chain_id,
        **llm_chain_kwargs
    )

    llm_model_id = uuid.uuid4()
    sample_chat_messages = [
        [
            SystemMessage(
                content="You are a helpful assistant that translates English to French",
                additional_kwargs={},
            ),
            HumanMessage(
                content="I love programming.", additional_kwargs={}, example=False
            ),
        ]
    ]
    event_dispatcher.on_chat_model_start(
        serialized=chat_serialized_data,
        messages=sample_chat_messages,
        run_id=llm_model_id,
        parent_run_id=llm_chain_id,
        invocation_params=chat_invocation_params,
        **default_kwargs
    )

    assert event_dispatcher.events_storage.events is not None
    assert len(event_dispatcher.events_storage.events) == 3
    assert llm_model_id in event_dispatcher.events_storage.events
    assert event_dispatcher.events_storage.events[llm_model_id].event_id == llm_model_id
    assert event_dispatcher.events_storage.events[llm_model_id].hierarchy is not None
    assert event_dispatcher.events_storage.events[llm_model_id].hierarchy.parent_run_id == llm_chain_id  # type: ignore # pylint: disable=line-too-long
    assert event_dispatcher.events_storage.events[llm_model_id].data == EventData(
        type=EventType.CHAT_MODEL,
        serialized=chat_serialized_data,
        inputs={"messages": sample_chat_messages},
        input_extras=dict(
            {"invocation_params": chat_invocation_params}, **default_kwargs
        ),
    )


def test_event_pairing_dispatcher_on_llm_end(  # pylint: disable=too-many-arguments
    simple_sequential_chain_serialized_data: dict[str, Any],
    llm_chain_serialized_data: dict[str, Any],
    llm_serialized_data: dict[str, Any],
    llm_invocation_params: dict[str, Any],
    default_kwargs: dict[str, Any],
    llm_chain_kwargs: dict[str, Any],
) -> None:
    chain_id = uuid.uuid4()
    event_dispatcher = EventPairingDispatcher()
    event_dispatcher.on_chain_start(
        serialized=simple_sequential_chain_serialized_data,
        inputs={"era": "Victorian England", "title": "Tragedy at sunset on the beach"},
        run_id=chain_id,
        parent_run_id=None,
        **default_kwargs
    )

    llm_chain_id = uuid.uuid4()
    event_dispatcher.on_chain_start(
        serialized=llm_chain_serialized_data,
        inputs={"era": "Victorian England", "title": "Tragedy at sunset on the beach"},
        run_id=llm_chain_id,
        parent_run_id=chain_id,
        **llm_chain_kwargs
    )

    llm_model_id = uuid.uuid4()
    sample_llm_prompts = [
        "You are a playwright. Given the title of play and the era it is set in, "
        "it is your job to write a synopsis for that title.\n\nTitle: Tragedy at "
        "sunset on the beach\nEra: Victorian England\nPlaywright: This is a synopsis "
        "for the above play:"
    ]
    event_dispatcher.on_llm_start(
        serialized=llm_serialized_data,
        prompts=sample_llm_prompts,
        run_id=llm_model_id,
        parent_run_id=llm_chain_id,
        invocation_params=llm_invocation_params,
        **default_kwargs
    )

    llm_model_sample_output = LLMResult(
        generations=[
            [
                Generation(
                    text="\n\nThe play is set in Victorian England and"
                    " follows the story of a young family...",
                    generation_info={"finish_reason": "stop", "logprobs": None},
                )
            ]
        ],
        llm_output={
            "model_name": "text-davinci-003",
            "token_usage": {
                "completion_tokens": 152,
                "prompt_tokens": 62,
                "total_tokens": 214,
            },
        },
    )

    watched_event = event_dispatcher.on_llm_end(
        response=llm_model_sample_output,
        parent_run_id=llm_chain_id,
        run_id=llm_model_id,
        tags=[],
    )

    assert event_dispatcher.events_storage.events is not None
    assert len(event_dispatcher.events_storage.events) == 3
    assert watched_event is not None
    assert watched_event.run_id == llm_model_id
    assert watched_event.hierarchy is not None
    assert watched_event.hierarchy.parent_run_id == llm_chain_id
    assert watched_event.hierarchy.root_run_id == chain_id
    assert watched_event.inputs == {"prompts": sample_llm_prompts}
    assert watched_event.outputs == {"response": llm_model_sample_output}
    assert watched_event.type is EventType.LLM_MODEL
    assert watched_event.serialized == llm_serialized_data
    assert watched_event.extras is not None
    assert watched_event.extras.input == dict(
        {"invocation_params": llm_invocation_params}, **default_kwargs
    )
    assert watched_event.extras.output == {"tags": []}


def test_event_pairing_dispatcher_on_llm_end__chat_model(  # pylint: disable=too-many-arguments
    simple_sequential_chain_serialized_data: dict[str, Any],
    llm_chain_serialized_data: dict[str, Any],
    chat_serialized_data: dict[str, Any],
    chat_invocation_params: dict[str, Any],
    default_kwargs: dict[str, Any],
    llm_chain_kwargs: dict[str, Any],
) -> None:
    chain_id = uuid.uuid4()
    event_dispatcher = EventPairingDispatcher()
    event_dispatcher.on_chain_start(
        serialized=simple_sequential_chain_serialized_data,
        inputs={"era": "Victorian England", "title": "Tragedy at sunset on the beach"},
        run_id=chain_id,
        parent_run_id=None,
        **default_kwargs
    )

    llm_chain_id = uuid.uuid4()
    event_dispatcher.on_chain_start(
        serialized=llm_chain_serialized_data,
        inputs={"era": "Victorian England", "title": "Tragedy at sunset on the beach"},
        run_id=llm_chain_id,
        parent_run_id=chain_id,
        **llm_chain_kwargs
    )

    chat_model_id = uuid.uuid4()
    sample_chat_messages = [
        [
            SystemMessage(
                content="You are a helpful assistant that translates English to French",
                additional_kwargs={},
            ),
            HumanMessage(
                content="I love programming.", additional_kwargs={}, example=False
            ),
        ]
    ]
    event_dispatcher.on_chat_model_start(
        serialized=chat_serialized_data,
        messages=sample_chat_messages,
        run_id=chat_model_id,
        parent_run_id=llm_chain_id,
        invocation_params=chat_invocation_params,
        **default_kwargs
    )

    chat_model_sample_output = LLMResult(
        generations=[
            [
                ChatGeneration(
                    text="J'adore la programmation.",
                    generation_info={"finish_reason": "stop"},
                    message=AIMessage(
                        content="J'adore la programmation.",
                        additional_kwargs={},
                        example=False,
                    ),
                )
            ]
        ],
        llm_output={
            "model_name": "gpt-3.5-turbo",
            "token_usage": {
                "prompt_tokens": 26,
                "completion_tokens": 8,
                "total_tokens": 34,
            },
        },
    )

    watched_event = event_dispatcher.on_llm_end(
        response=chat_model_sample_output,
        parent_run_id=llm_chain_id,
        run_id=chat_model_id,
        tags=[],
    )

    assert event_dispatcher.events_storage.events is not None
    assert len(event_dispatcher.events_storage.events) == 3
    assert watched_event is not None
    assert watched_event.run_id == chat_model_id
    assert watched_event.hierarchy is not None
    assert watched_event.hierarchy.parent_run_id == llm_chain_id
    assert watched_event.hierarchy.root_run_id == chain_id
    assert watched_event.inputs == {"messages": sample_chat_messages}
    assert watched_event.outputs == {"response": chat_model_sample_output}
    assert watched_event.type is EventType.CHAT_MODEL
    assert watched_event.serialized == chat_serialized_data
    assert watched_event.extras is not None
    assert watched_event.extras.input == dict(
        {"invocation_params": chat_invocation_params}, **default_kwargs
    )
    assert watched_event.extras.output == {"tags": []}
