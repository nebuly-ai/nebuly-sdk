import logging
from typing import Any

from langchain.callbacks.base import BaseCallbackHandler

from nebuly.event_pairing_dispatchers import EventPairingDispatcher

logger = logging.getLogger(__name__)


class LangChainTrackingHandler(BaseCallbackHandler):
    def __init__(self, event_pairing_dispatcher: EventPairingDispatcher):
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
