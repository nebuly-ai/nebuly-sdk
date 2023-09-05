from nebuly.patcher import Watched


class Nebuly:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.watched: list[Watched] = []

    def observer(self, watched: Watched) -> None:
        self.watched.append(watched)
