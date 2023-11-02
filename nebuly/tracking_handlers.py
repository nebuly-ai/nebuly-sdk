from __future__ import annotations

import logging
from typing import Any, Sequence
from uuid import UUID

from langchain.callbacks.base import BaseCallbackHandler, BaseCallbackManager, Callbacks
from langchain.chains.base import Chain
from langchain.schema import Document
from langchain.schema.messages import BaseMessage
from langchain.schema.output import LLMResult
from langchain.schema.runnable.base import RunnableParallel, RunnableSequence
from langchain.schema.runnable.config import RunnableConfig, ensure_config
from langchain.schema.runnable.utils import Input

from nebuly.contextmanager import (
    EventData,
    EventsStorage,
    InteractionContext,
    get_nearest_open_interaction,
)
from nebuly.entities import EventType
from nebuly.exceptions import NotInInteractionContext

logger = logging.getLogger(__name__)


def set_tracking_handlers() -> None:
    tracking_handler = LangChainTrackingHandler()
    original_call = Chain.__call__
    original_acall = Chain.acall
    original_sequence_invoke = RunnableSequence.invoke
    original_parallel_invoke = RunnableParallel.invoke
    original_sequence_ainvoke = RunnableSequence.ainvoke
    original_parallel_ainvoke = RunnableParallel.ainvoke
    original_sequence_stream = RunnableSequence.stream
    original_parallel_stream = RunnableParallel.stream
    original_sequence_astream = RunnableSequence.astream
    original_parallel_astream = RunnableParallel.astream

    def _add_tracking_handler(handlers: list) -> Callbacks:
        handlers_new = [tracking_handler]
        for callback in handlers:
            if not isinstance(callback, LangChainTrackingHandler):
                handlers_new.append(callback)
        return handlers_new

    def set_callbacks_arg(callbacks: Callbacks) -> Callbacks:
        if callbacks is None:
            callbacks = [tracking_handler]
        elif isinstance(callbacks, list):
            callbacks = _add_tracking_handler(callbacks)
        elif isinstance(callbacks, BaseCallbackManager):
            callbacks.handlers = _add_tracking_handler(callbacks.handlers)
            callbacks.inheritable_handlers = _add_tracking_handler(
                callbacks.inheritable_handlers
            )
        return callbacks

    def set_config_arg(config: RunnableConfig | None) -> RunnableConfig:
        config = ensure_config(config)
        config["callbacks"] = set_callbacks_arg(config["callbacks"])
        return config

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

    async def tracked_acall(
        self: Any,
        inputs: dict[str, Any],
        return_only_outputs: bool = False,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Any:
        callbacks = set_callbacks_arg(callbacks)
        tracking_handler.nebuly_user = kwargs.pop("user_id", None)
        tracking_handler.nebuly_user_group = kwargs.pop("user_group_profile", None)
        return await original_acall(
            self,
            inputs=inputs,
            return_only_outputs=return_only_outputs,
            callbacks=callbacks,
            **kwargs,
        )

    def tracked_invoke(
        self: Any, input: Input, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Any:
        config = set_config_arg(config)
        user_id = kwargs.pop("user_id", None)
        if user_id is not None:
            tracking_handler.nebuly_user = user_id
        user_group_profile = kwargs.pop("user_group_profile", None)
        if user_group_profile is not None:
            tracking_handler.nebuly_user_group = user_group_profile

        return (
            original_sequence_invoke(
                self,
                input=input,
                config=config,
            )
            if isinstance(self, RunnableSequence)
            else original_parallel_invoke(
                self,
                input=input,
                config=config,
            )
        )

    async def tracked_ainvoke(
        self: Any, input: Input, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Any:
        config = set_config_arg(config)
        user_id = kwargs.pop("user_id", None)
        if user_id is not None:
            tracking_handler.nebuly_user = user_id
        user_group_profile = kwargs.pop("user_group_profile", None)
        if user_group_profile is not None:
            tracking_handler.nebuly_user_group = user_group_profile
        return (
            await original_sequence_ainvoke(
                self,
                input=input,
                config=config,
            )
            if isinstance(self, RunnableSequence)
            else await original_parallel_ainvoke(
                self,
                input=input,
                config=config,
            )
        )

    def tracked_stream(
        self: Any, input: Input, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Any:
        config = set_config_arg(config)
        user_id = kwargs.pop("user_id", None)
        if user_id is not None:
            tracking_handler.nebuly_user = user_id
        user_group_profile = kwargs.pop("user_group_profile", None)
        if user_group_profile is not None:
            tracking_handler.nebuly_user_group = user_group_profile

        if isinstance(self, RunnableSequence):
            return original_sequence_stream(
                self,
                input=input,
                config=config,
            )
        return original_parallel_stream(
            self,
            input=input,
            config=config,
        )

    async def tracked_astream(
        self: Any, input: Input, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Any:
        config = set_config_arg(config)
        user_id = kwargs.pop("user_id", None)
        if user_id is not None:
            tracking_handler.nebuly_user = user_id
        user_group_profile = kwargs.pop("user_group_profile", None)
        if user_group_profile is not None:
            tracking_handler.nebuly_user_group = user_group_profile
        if isinstance(self, RunnableSequence):
            async for chunk in await original_sequence_astream(
                self,
                input=input,
                config=config,
            ):
                yield chunk
        else:
            async for chunk in await original_parallel_astream(
                self,
                input=input,
                config=config,
            ):
                yield chunk

    Chain.__call__ = tracked_call  # type: ignore
    Chain.acall = tracked_acall  # type: ignore
    RunnableSequence.invoke = tracked_invoke  # type: ignore
    RunnableParallel.invoke = tracked_invoke  # type: ignore
    RunnableSequence.ainvoke = tracked_ainvoke  # type: ignore
    RunnableParallel.ainvoke = tracked_ainvoke  # type: ignore
    RunnableSequence.stream = tracked_stream  # type: ignore
    RunnableParallel.stream = tracked_stream  # type: ignore
    RunnableSequence.astream = tracked_astream  # type: ignore
    RunnableParallel.astream = tracked_astream  # type: ignore


class LangChainTrackingHandler(BaseCallbackHandler):  # noqa
    def __init__(self) -> None:
        self.nebuly_user = None
        self.nebuly_user_group = None
        self._current_interaction: InteractionContext | None = None

    @property
    def current_interaction_storage(self) -> EventsStorage:
        return (
            self.current_interaction._events_storage  # pylint: disable=protected-access  # noqa: E501
        )

    @property
    def current_interaction(self) -> InteractionContext:
        try:
            self._current_interaction = get_nearest_open_interaction()
        except NotInInteractionContext:
            if self._current_interaction is None:
                raise NotInInteractionContext(
                    "The current interaction is not set and there "
                    "is no open interaction"
                )
            return self._current_interaction
        return self._current_interaction

    def set_interaction(self, interaction: InteractionContext) -> None:
        self._current_interaction = interaction

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
