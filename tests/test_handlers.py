from nebuly.event_pairing_dispatchers import LangChainEventPairingDispatcher
from nebuly.handlers import LangChainTrackingHandler


def test_langchain_tracking_handler__can_instantiate() -> None:
    event_pairing_dispatcher = LangChainEventPairingDispatcher()
    tracking_handler = LangChainTrackingHandler(
        event_pairing_dispatcher=event_pairing_dispatcher
    )
    assert tracking_handler is not None
