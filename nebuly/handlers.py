import logging
from typing import Any

from langchain.callbacks.base import BaseCallbackHandler, BaseCallbackManager, Callbacks
from langchain.chains.base import Chain

from nebuly.event_pairing_dispatchers import LangChainEventPairingDispatcher

logger = logging.getLogger(__name__)


def set_tracking_handlers() -> None:
    event_pairing_dispatcher = LangChainEventPairingDispatcher()
    tracking_handler = LangChainTrackingHandler(
        event_pairing_dispatcher=event_pairing_dispatcher
    )
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
        inputs: dict[str, Any],
        return_only_outputs: bool = False,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Any:
        callbacks = set_callbacks_arg(callbacks)
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
    def __init__(self, event_pairing_dispatcher: LangChainEventPairingDispatcher):
        self.event_pairing_dispatcher = event_pairing_dispatcher

    def on_retriever_start(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        logger.debug("Called on_retriever_start")
        self.event_pairing_dispatcher.on_retriever_start(*args, **kwargs)

    def on_retriever_end(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        logger.debug("Called on_retriever_end")
        self.event_pairing_dispatcher.on_retriever_end(*args, **kwargs)

    def on_tool_start(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        logger.debug("Called on_tool_start")
        self.event_pairing_dispatcher.on_tool_start(*args, **kwargs)

    def on_tool_end(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        logger.debug("Called on_tool_end")
        self.event_pairing_dispatcher.on_tool_end(*args, **kwargs)

    def on_llm_start(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        logger.debug("Called on_llm_start")
        self.event_pairing_dispatcher.on_llm_start(*args, **kwargs)

    def on_llm_end(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        logger.debug("Called on_llm_end")
        self.event_pairing_dispatcher.on_llm_end(*args, **kwargs)

    def on_chat_model_start(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        logger.debug("Called on_chat_model_start")
        self.event_pairing_dispatcher.on_chat_model_start(*args, **kwargs)

    def on_chain_start(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        logger.debug("Called on_chain_start")
        self.event_pairing_dispatcher.on_chain_start(*args, **kwargs)

    def on_chain_end(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        logger.debug("Called on_chain_end")
        self.event_pairing_dispatcher.on_chain_end(*args, **kwargs)
