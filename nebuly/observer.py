from nebuly.entities import Message, Publisher_T, Watched


class NebulyObserver:
    """
    NebulyObserver is an observer that sends a message to the API when a
    patched function is called.
    """

    def __init__(
        self, *, api_key: str, project: str, phase: str, publish: Publisher_T
    ) -> None:
        self.api_key = api_key
        self.project = project
        self.phase = phase
        self.publisher = publish

    def observe(self, watched: Watched) -> None:
        message = Message(
            api_key=self.api_key,
            project=self.project,
            phase=self.phase,
            watched=watched,
        )
        self.publisher(message)
