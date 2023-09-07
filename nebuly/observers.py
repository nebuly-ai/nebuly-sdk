from nebuly.entities import Message, Publisher_T, Watched


class NebulyObserver:
    """
    NebulyObserver is an observer that sends a message to the API when a
    patched function is called.
    """

    def __init__(
        self,
        *,
        api_key: str,
        project: str | None,
        phase: str | None,
        publish: Publisher_T
    ) -> None:
        self._api_key = api_key
        self._project = project
        self._phase = phase
        self._publisher = publish

    def _set_nebuly_kwargs(self, watched: Watched) -> None:
        if "nebuly_project" not in watched.called_with_nebuly_kwargs and self._project:
            watched.called_with_nebuly_kwargs["nebuly_project"] = self._project
        if "nebuly_phase" not in watched.called_with_nebuly_kwargs and self._phase:
            watched.called_with_nebuly_kwargs["nebuly_phase"] = self._phase

    def on_event_received(self, watched: Watched) -> None:
        self._set_nebuly_kwargs(watched)
        message = Message(
            api_key=self._api_key,
            watched=watched,
        )
        self._publisher(message)
