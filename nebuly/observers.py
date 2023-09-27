from __future__ import annotations

from nebuly.entities import InteractionWatch, Publisher


class NebulyObserver:
    """
    NebulyObserver is an observer that sends a message to the API when a
    patched function is called.
    """

    def __init__(
        self,
        *,
        api_key: str,
        publish: Publisher,
    ) -> None:
        self._api_key = api_key
        self._publisher = publish

    def on_event_received(self, interaction_watch: InteractionWatch) -> None:
        self._publisher(interaction_watch)
