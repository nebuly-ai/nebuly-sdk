from __future__ import annotations

import logging
from typing import Any, Sequence
from uuid import UUID

from langchain.callbacks.base import BaseCallbackHandler, BaseCallbackManager, Callbacks
from langchain.chains.base import Chain
from langchain.schema import Document
from langchain.schema.messages import BaseMessage
from langchain.schema.output import LLMResult

from nebuly.contextmanager import EventData, EventsStorage, get_nearest_open_interaction
from nebuly.entities import EventType

logger = logging.getLogger(__name__)


def set_tracking_handlers() -> None:
    tracking_handler = LangChainTrackingHandler()
    original_call = Chain.__call__
    original_acall = Chain.acall

    def set_callbacks_arg(callbacks: Callbacks) -> Callbacks:
        if callbacks is None:
            callbacks = [tracking_handler]
        elif isinstance(callbacks, list):
            if tracking_handler not in callbacks:
                callbacks.append(tracking_handler)
        elif isinstance(callbacks, BaseCallbackManager):
            if tracking_handler not in callbacks.handlers:
                callbacks.handlers.append(tracking_handler)
            if tracking_handler not in callbacks.inheritable_handlers:
                callbacks.inheritable_handlers.append(tracking_handler)
        return callbacks

    def tracked_call(
        self: Any,
        inputs: dict[str, Any] | Any,
        return_only_outputs: bool = False,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Any:
        callbacks = set_callbacks_arg(callbacks)
        if isinstance(inputs, dict):
            tracking_handler.nebuly_user = inputs.pop("user_id", None)
            tracking_handler.nebuly_user_group = inputs.pop("user_group_profile", None)
        if tracking_handler.nebuly_user is None:
            tracking_handler.nebuly_user = kwargs.pop("user_id", None)
        if tracking_handler.nebuly_user_group is None:
            tracking_handler.nebuly_user_group = kwargs.pop("user_group_profile", None)
        return original_call(
            self,
            inputs=inputs,
            return_only_outputs=return_only_outputs,
            callbacks=callbacks,
            **kwargs,
        )

    def tracked_acall(
        self: Any,
        inputs: dict[str, Any],
        return_only_outputs: bool = False,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Any:
        callbacks = set_callbacks_arg(callbacks)
        tracking_handler.nebuly_user = kwargs.pop("user_id", None)
        tracking_handler.nebuly_user_group = kwargs.pop("user_group_profile", None)
        return original_acall(
            self,
            inputs=inputs,
            return_only_outputs=return_only_outputs,
            callbacks=callbacks,
            **kwargs,
        )

    Chain.__call__ = tracked_call  # type: ignore
    Chain.acall = tracked_acall  # type: ignore


class LangChainTrackingHandler(BaseCallbackHandler):  # noqa
    def __init__(self) -> None:
        self.nebuly_user = None
        self.nebuly_user_group = None

    @property
    def current_interaction_storage(self) -> EventsStorage:
        return (
            get_nearest_open_interaction()._events_storage  # pylint: disable=protected-access  # noqa: E501
        )

    def on_tool_start(  # pylint: disable=arguments-differ
        self,
        serialized: dict[str, Any],
        input_str: str,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        data = EventData(
            type=EventType.TOOL,
            kwargs={
                "serialized": serialized,
                "input_str": input_str,
                **kwargs,
            },
        )
        self.current_interaction_storage.add_event(
            run_id, parent_run_id, data, module="langchain"
        )

    def on_tool_end(  # pylint: disable=arguments-differ
        self,
        output: str,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self.current_interaction_storage.events[run_id].data.add_end_event_data(
            kwargs=kwargs, output=output
        )
        self.current_interaction_storage.events[run_id].set_end_time()

    def on_retriever_start(  # pylint: disable=arguments-differ
        self,
        serialized: dict[str, Any],
        query: str,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        data = EventData(
            type=EventType.RETRIEVAL,
            kwargs={
                "serialized": serialized,
                "query": query,
                **kwargs,
            },
        )
        self.current_interaction_storage.add_event(
            run_id, parent_run_id, data, module="langchain"
        )

    def on_retriever_end(  # pylint: disable=arguments-differ
        self,
        documents: Sequence[Document],
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self.current_interaction_storage.events[run_id].data.add_end_event_data(
            kwargs=kwargs, output=documents
        )
        self.current_interaction_storage.events[run_id].set_end_time()

    def on_llm_start(  # pylint: disable=arguments-differ
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        data = EventData(
            type=EventType.LLM_MODEL,
            kwargs={
                "serialized": serialized,
                "prompts": prompts,
                **kwargs,
            },
        )
        self.current_interaction_storage.add_event(
            run_id, parent_run_id, data, module="langchain"
        )

    def on_llm_end(  # pylint: disable=arguments-differ
        self,
        response: LLMResult,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self.current_interaction_storage.events[run_id].data.add_end_event_data(
            kwargs=kwargs, output=response
        )
        self.current_interaction_storage.events[run_id].set_end_time()

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
        self.current_interaction_storage.add_event(
            run_id, parent_run_id, data, module="langchain"
        )

    def on_chain_start(  # pylint: disable=arguments-differ
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        data = EventData(
            type=EventType.CHAIN,
            kwargs={
                "serialized": serialized,
                "inputs": inputs,
                **kwargs,
            },
        )
        self.current_interaction_storage.add_event(
            run_id, parent_run_id, data, module="langchain"
        )

    def on_chain_end(  # pylint: disable=arguments-differ
        self,
        outputs: dict[str, Any],
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self.current_interaction_storage.events[run_id].data.add_end_event_data(
            kwargs=kwargs, output=outputs
        )
        self.current_interaction_storage.events[run_id].set_end_time()
