from nebuly.handlers import LangChainTrackingHandler


def test_langchain_tracking_handler__can_instantiate() -> None:
    tracking_handler = LangChainTrackingHandler()
    assert tracking_handler is not None
