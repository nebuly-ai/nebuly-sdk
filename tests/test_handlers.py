from nebuly.event_pairing_dispatchers import LangChainEventPairingDispatcher
from nebuly.handlers import LangChainTrackingHandler
from tests.test_monkey_patching import Observer


def test_langchain_tracking_handler__can_instantiate() -> None:
    observer = Observer()
    event_pairing_dispatcher = LangChainEventPairingDispatcher(observer)
    tracking_handler = LangChainTrackingHandler(
        event_pairing_dispatcher=event_pairing_dispatcher
    )
    assert tracking_handler is not None
