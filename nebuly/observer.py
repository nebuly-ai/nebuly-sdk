from typing import Callable

from nebuly.entities import Message, Watched


Publisher_T = Callable[[Message], None]


class NebulyObserver:
    def __init__(
        self, *, api_key: str, project: str, phase: str, publisher: Publisher_T
    ) -> None:
        self.api_key = api_key
        self.project = project
        self.phase = phase
        self.publisher = publisher

    def observe(self, watched: Watched) -> None:
        message = Message(
            api_key=self.api_key,
            project=self.project,
            phase=self.phase,
            watched=watched,
        )
        self.publisher(message)
