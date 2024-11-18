from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Sequence, Tuple, cast
from uuid import UUID

import langchain
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains.base import Chain
from langchain.load import load
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.prompts.chat import AIMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import Document
from langchain.schema.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain.schema.output import LLMResult
from langchain.schema.runnable import RunnableSequence

from nebuly.entities import HistoryEntry, InteractionWatch, ModelInput, SpanWatch
from nebuly.events import Event, EventData, EventsStorage
from nebuly.requests import post_message

logger = logging.getLogger(__name__)


def _get_spans(events: dict[UUID, Event]) -> list[SpanWatch]:
    return [event.as_span_watch() for event in events.values()]


def _process_prompt_template(
    inputs: dict[str, Any] | Any, prompt: PromptTemplate
) -> str:
    if isinstance(inputs, dict):
        return prompt.format(**{key: inputs.get(key) for key in prompt.input_variables})
    return prompt.format(**{prompt.input_variables[0]: inputs})


def _process_chat_prompt_template(
    inputs: dict[str, Any] | Any, prompt: ChatPromptTemplate
) -> tuple[str, list[HistoryEntry]]:
    messages = []
    for message in prompt.messages:
        if isinstance(inputs, dict):
            input_vars = {key: inputs.get(key) for key in message.input_variables}  # type: ignore  # noqa: E501  # pylint: disable=line-too-long
        else:
            input_vars = {message.input_variables[0]: inputs}  # type: ignore
        if isinstance(message, (HumanMessagePromptTemplate, AIMessagePromptTemplate)):
            messages.append((message.format(**input_vars).content))
    last_prompt = str(messages[-1])
    message_history = messages[:-1]

    if len(message_history) % 2 != 0:
        logger.warning("Odd number of chat history elements, ignoring last element")
        message_history = message_history[:-1]

    # Convert the history to [(user, assistant), ...] format
    history = [
        HistoryEntry(
            user=str(message_history[i]), assistant=str(message_history[i + 1])
        )
        for i in range(0, len(message_history), 2)
        if i < len(message_history) - 1
    ]

    return last_prompt, history


def _get_input_and_history(chain: Chain, inputs: dict[str, Any] | Any) -> ModelInput:
    chains = getattr(chain, "chains", None)
    if chains is not None:
        # If the chain is a SequentialChain, we need to get the
        # prompt from the first chain
        prompt = getattr(chains[0], "prompt", None)
    else:
        # If the chain is not a SequentialChain, we need to get
        # the prompt from the chain
        prompt = getattr(chain, "prompt", None)
        if prompt is None:
            if not isinstance(inputs, dict):
                return ModelInput(prompt=inputs)
            return ModelInput(prompt=inputs["input"])

    if isinstance(prompt, PromptTemplate):
        return ModelInput(prompt=_process_prompt_template(inputs, prompt))

    if isinstance(prompt, ChatPromptTemplate):
        prompt, history = _process_chat_prompt_template(inputs, prompt)
        return ModelInput(prompt=prompt, history=history)

    raise ValueError(f"Unknown prompt type: {prompt}")


def _get_input_and_history_runnable_seq(
    sequence: RunnableSequence[Any, Any], inputs: dict[str, Any] | Any
) -> ModelInput:
    steps = getattr(sequence, "steps", None)

    if steps is not None and len(steps) > 0:
        for step in steps:
            if isinstance(step, PromptTemplate):
                return ModelInput(prompt=_process_prompt_template(inputs, step))

            if isinstance(step, ChatPromptTemplate):
                prompt, history = _process_chat_prompt_template(inputs, step)
                return ModelInput(prompt=prompt, history=history)

    return ModelInput(prompt="")


def _get_output_chain(chain: Chain, result: dict[str, Any]) -> str:
    if len(chain.output_keys) == 1:
        return str(result[chain.output_keys[0]])
    output = {}
    for key in chain.output_keys:
        output[key] = result[key]
    return _parse_output(output)


def _parse_output(output: str | dict[str, Any] | AIMessage) -> str:
    if isinstance(output, dict):
        if "answer" in output:
            return str(output["answer"])
        if "output" in output:
            return str(output["output"])
        return "\n".join([f"{key}: {value}" for key, value in output.items()])
    if isinstance(output, AIMessage):
        return str(output.content)
    return output


def _parse_langchain_data(  # pylint: disable=too-many-return-statements
    data: Any,
) -> str:
    if isinstance(data, dict):
        if len(data) == 1 and isinstance(list(data.values())[0], str):
            return str(list(data.values())[0])
        if "answer" in data:
            # If the data is a retrieval chain, we want to return the answer
            return str(data["answer"])
        if "output" in data:
            # If the data is a retrieval chain, we want to return the output
            return str(data["output"])
        if "input" in data:
            # Needed when calling invoke with input and chat_history
            return str(data["input"])
        if "question" in data:
            # Needed when calling invoke with question and chat_history
            return str(data["question"])
        if "messages" in data and isinstance(
            data["messages"][-1], (HumanMessage, AIMessage, ToolMessage)
        ):
            return str(data["messages"][-1].content)
        return "\n".join([f"{key}: {value}" for key, value in data.items()])
    if isinstance(data, list):
        data = data[-1]
    if isinstance(data, (HumanMessage, AIMessage, ToolMessage)):
        return str(data.content)
    return str(data)


def _parse_langchain_history(inputs: dict[str, Any]) -> list[HistoryEntry]:
    if "chat_history" in inputs:
        history = inputs["chat_history"]
        if len(history) > 0:
            if isinstance(history[0], tuple):
                # In some chains, the chat history is a list of tuples
                return [
                    HistoryEntry(user=str(message[0]), assistant=str(message[1]))
                    for message in history
                ]

            human_messages = [message for message in history if message.type == "human"]
            ai_messages = [message for message in history if message.type == "ai"]

            if len(human_messages) != len(ai_messages):
                logger.warning(
                    "Unequal number of human and AI messages in chat history"
                )

            return [
                HistoryEntry(user=str(human.content), assistant=str(ai.content))
                for human, ai in zip(human_messages, ai_messages)
            ]
    return []


class EventType(Enum):
    """
    The type of event generated by LangChain.
    """

    CHAIN = "chain"
    TOOL = "tool"
    RETRIEVAL = "retrieval"
    LLM_MODEL = "llm_model"
    CHAT_MODEL = "chat_model"


@dataclass
class LangChainEvent(Event):
    input_llm = None

    @property
    def input(self) -> str:  # pylint: disable=too-many-return-statements
        if self.input_llm is not None:
            return self.input_llm
        if self.data.kwargs is None:
            raise ValueError("Event has no kwargs.")
        if self.data.type is EventType.CHAT_MODEL:
            messages = self.data.kwargs.get("messages")
            if messages is None:
                raise ValueError("Event has no messages.")
            return str(messages[-1][-1].content)
        if self.data.type is EventType.LLM_MODEL:
            prompts = self.data.kwargs.get("prompts")
            if prompts is None:
                raise ValueError("Event has no prompts.")
            return str(prompts[-1])
        if self.data.type is EventType.CHAIN:
            inputs = self.data.kwargs.get("inputs")
            if inputs is None:
                raise ValueError("Event has no inputs.")
            try:
                chain = load(self.data.kwargs.get("serialized", {}))
                if chain is None:
                    return _parse_langchain_data(inputs)
                if isinstance(chain, RunnableSequence):
                    return _get_input_and_history_runnable_seq(chain, inputs).prompt
                return _get_input_and_history(chain, inputs).prompt
            except NotImplementedError:
                return _parse_langchain_data(inputs)

        raise ValueError(f"Event type {self.data.type} not supported.")

    @property
    def output(self) -> str:
        if self.data.output is None:
            raise ValueError("Event has no output.")
        if self.data.type in [EventType.CHAT_MODEL, EventType.LLM_MODEL]:
            return str(self.data.output.generations[-1][0].text)
        if self.data.type is EventType.CHAIN:
            if self.data.kwargs is None:
                raise ValueError("Event has no kwargs.")
            try:
                chain = load(self.data.kwargs.get("serialized", {}))
                if chain is None:
                    return _parse_langchain_data(self.data.output)
                if isinstance(chain, RunnableSequence):
                    return _parse_output(self.data.output)
                return _get_output_chain(chain, self.data.output)
            except NotImplementedError:
                return _parse_langchain_data(self.data.output)

        raise ValueError(f"Event type {self.data.type} not supported.")

    @property
    def history(  # pylint: disable=too-many-return-statements
        self,
    ) -> list[HistoryEntry]:
        if self.data.kwargs is None:
            raise ValueError("Event has no kwargs.")

        if self.data.type is EventType.CHAT_MODEL:
            messages = self.data.kwargs.get("messages")
            if messages is None:
                raise ValueError("Event has no messages.")
            history = [
                message
                for message in messages[-1][:-1]
                if isinstance(message, (HumanMessage, AIMessage))
            ]
            if len(history) % 2 != 0:
                raise ValueError(
                    "Odd number of chat history elements, please provide "
                    "a valid history."
                )
            if len(history) == 0:
                return []
            return [
                HistoryEntry(
                    user=str(history[i].content), assistant=str(history[i + 1].content)
                )
                for i in range(0, len(history) - 1, 2)
            ]
        if self.data.type is EventType.LLM_MODEL:
            return []
        if self.data.type is EventType.CHAIN:
            inputs = self.data.kwargs.get("inputs")
            if inputs is None:
                raise ValueError("Event has no inputs.")
            try:
                chain = load(self.data.kwargs.get("serialized", {}))
                if chain is None:
                    return _parse_langchain_history(inputs)
                if isinstance(chain, RunnableSequence):
                    return _get_input_and_history_runnable_seq(chain, inputs).history
                return _get_input_and_history(chain, inputs).history
            except NotImplementedError:
                return _parse_langchain_history(inputs)

        raise ValueError(f"Event type {self.data.type} not supported.")

    def as_span_watch(self) -> SpanWatch:
        if self.end_time is None:
            raise ValueError("Event has not been finished yet.")
        return SpanWatch(
            span_id=self.event_id,
            module=self.module,
            version="unknown",
            function=self._get_function(),
            called_start=self.start_time,
            called_end=self.end_time,
            called_with_args=cast(Tuple[Any], self.data.args),
            called_with_kwargs=cast(Dict[str, Any], self.data.kwargs),
            returned=self.data.output,
            generator=False,
            generator_first_element_timestamp=None,
            provider_extras={
                "event_type": self.data.type.value,
            },
            rag_source=self._get_rag_source(),
        )

    def _get_function(self) -> str:
        if self.data.kwargs is None or len(self.data.kwargs) == 0:
            raise ValueError("Event has no kwargs.")
        if self.data.type is EventType.TOOL:
            return self.data.kwargs["serialized"]["name"]  # type: ignore
        try:
            return ".".join(self.data.kwargs["serialized"]["id"])
        except TypeError:
            return "unknown"

    def _get_rag_source(self) -> str | None:
        if self.data.kwargs is None or len(self.data.kwargs) == 0:
            raise ValueError("Event has no kwargs.")
        if self.data.type is EventType.TOOL:
            serialized = self.data.kwargs["serialized"]
            if serialized is None and "name" in self.data.kwargs:
                return str(self.data.kwargs["name"])
            return self.data.kwargs["serialized"]["name"]  # type: ignore
        if self.data.type is EventType.RETRIEVAL:
            serialized = self.data.kwargs["serialized"]
            if serialized is None and "name" in self.data.kwargs:
                return str(self.data.kwargs["name"])
            return self.data.kwargs["serialized"]["id"][-1]  # type: ignore

        return None


# pylint: disable=too-many-instance-attributes
class LangChainTrackingHandler(BaseCallbackHandler):  # noqa
    def __init__(  # pylint: disable=too-many-arguments
        self,
        api_key: str,
        user_id: str,
        user_group_profile: str | None = None,
        feature_flags: list[str] | None = None,
        nebuly_tags: dict[str, str] | None = None,
        conversation_id: str | None = None,
        verbose: bool = False,
    ) -> None:
        self._check_version()
        self.api_key = api_key
        self.nebuly_user = user_id
        self.nebuly_user_group = user_group_profile
        self.nebuly_feature_flags = feature_flags
        self.nebuly_tags = nebuly_tags
        self.conversation_id = conversation_id
        self._events_storage = EventsStorage()
        self.verbose = verbose

    @staticmethod
    def _check_version() -> None:
        if langchain.__version__ < "0.1.0":
            logger.warning(
                "LangChain version 0.1.0 or higher is required to use the "
                "LangChainTrackingHandler with Nebuly SDK. "
                "Please upgrade your LangChain version.",
            )

    def _send_interaction(self, run_id: uuid.UUID) -> None:
        if (
            len(self._events_storage.events) < 1
            or run_id not in self._events_storage.events
        ):
            raise ValueError(f"Event {run_id} not found in events storage.")

        interaction = InteractionWatch(
            end_user=self.nebuly_user,
            end_user_group_profile=self.nebuly_user_group,
            input=self._events_storage.events[run_id].input,
            output=self._events_storage.events[run_id].output,
            time_start=self._events_storage.events[run_id].start_time,
            time_end=cast(datetime, self._events_storage.events[run_id].end_time),
            history=self._events_storage.events[run_id].history,
            spans=_get_spans(events=self._events_storage.events),
            conversation_id=self.conversation_id,
            hierarchy={
                event.event_id: (
                    event.hierarchy.parent_run_id
                    if event.hierarchy is not None
                    else None
                )
                for event in self._events_storage.events.values()
            },
            feature_flags=self.nebuly_feature_flags,
            tags=self.nebuly_tags,
        )
        post_message(interaction, self.api_key)

    def on_tool_start(  # pylint: disable=arguments-differ
        self,
        serialized: dict[str, Any],
        input_str: str,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        if self.verbose:
            logger.info("Tool started with input: %s", input_str)
        data = EventData(
            type=EventType.TOOL,
            kwargs={
                "serialized": serialized,
                "input_str": input_str,
                **kwargs,
            },
        )
        self._events_storage.add_event(
            LangChainEvent, run_id, parent_run_id, data, module="langchain"
        )

    def on_tool_end(  # pylint: disable=arguments-differ
        self,
        output: str,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        if self.verbose:
            logger.info("Tool ended with output: %s", output)
        self._events_storage.events[run_id].data.add_end_event_data(
            kwargs=kwargs, output=output
        )
        self._events_storage.events[run_id].set_end_time()

    def on_retriever_start(  # pylint: disable=arguments-differ
        self,
        serialized: dict[str, Any],
        query: str,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        if self.verbose:
            logger.info("Retriever started with query: %s", query)
        data = EventData(
            type=EventType.RETRIEVAL,
            kwargs={
                "serialized": serialized,
                "query": query,
                **kwargs,
            },
        )
        self._events_storage.add_event(
            LangChainEvent, run_id, parent_run_id, data, module="langchain"
        )

    def on_retriever_end(  # pylint: disable=arguments-differ
        self,
        documents: Sequence[Document],
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        if self.verbose:
            logger.info("Retriever ended with %d documents", len(documents))
        self._events_storage.events[run_id].data.add_end_event_data(
            kwargs=kwargs, output=documents
        )
        self._events_storage.events[run_id].set_end_time()

    def on_llm_start(  # pylint: disable=arguments-differ
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        if self.verbose:
            logger.info("LLM model started with %d prompts", len(prompts))

        if len(prompts) == 1:
            self._events_storage.update_chain_input(prompts[0])

        data = EventData(
            type=EventType.LLM_MODEL,
            kwargs={
                "serialized": serialized,
                "prompts": prompts,
                **kwargs,
            },
        )
        self._events_storage.add_event(
            LangChainEvent, run_id, parent_run_id, data, module="langchain"
        )

    def on_llm_end(  # pylint: disable=arguments-differ
        self,
        response: LLMResult,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        if self.verbose:
            logger.info("LLM model ended")
        self._events_storage.events[run_id].data.add_end_event_data(
            kwargs=kwargs, output=response
        )
        self._events_storage.events[run_id].set_end_time()

        if len(self._events_storage.events) == 1:
            if self.verbose:
                logger.info("Sending interaction for id: %s", str(run_id))
            self._send_interaction(run_id)
            self._events_storage.delete_events(run_id)

    def on_chat_model_start(  # pylint: disable=arguments-differ
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        data = EventData(
            type=EventType.CHAT_MODEL,
            kwargs={
                "serialized": serialized,
                "messages": messages,
                **kwargs,
            },
        )
        self._events_storage.add_event(
            LangChainEvent, run_id, parent_run_id, data, module="langchain"
        )

    def on_chain_start(  # pylint: disable=arguments-differ
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        if self.verbose:
            logger.info("Chain started with run_id: %s", str(run_id))
        data = EventData(
            type=EventType.CHAIN,
            kwargs={
                "serialized": serialized,
                "inputs": inputs,
                **kwargs,
            },
        )
        self._events_storage.add_event(
            LangChainEvent, run_id, parent_run_id, data, module="langchain"
        )

    def on_chain_end(  # pylint: disable=arguments-differ
        self,
        outputs: dict[str, Any],
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        if self.verbose:
            logger.info("Chain ended with run_id: %s", str(run_id))
        self._events_storage.events[run_id].data.add_end_event_data(
            kwargs=kwargs, output=outputs
        )
        self._events_storage.events[run_id].set_end_time()

        if self._events_storage.events[run_id].hierarchy is None:
            if self.verbose:
                logger.info("Sending interaction for id: %s", str(run_id))
            self._send_interaction(run_id)
            self._events_storage.delete_events(run_id)

    def send_pending_interaction(self, output: dict[str, str]) -> None:
        for (
            run_id
        ) in self._events_storage.events:  # pylint: disable=consider-using-dict-items
            if self._events_storage.events[run_id].hierarchy is None:
                self._events_storage.events[run_id].data.add_end_event_data(
                    kwargs={}, output=output
                )
                self._events_storage.events[run_id].set_end_time()
                # remove unfinised events
                for key, value in list(self._events_storage.events.items()):
                    if value.end_time is None:
                        self._events_storage.delete_events(key)
                self._send_interaction(run_id)
                self._events_storage.delete_events(run_id)
                break
