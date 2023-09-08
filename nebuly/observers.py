from __future__ import annotations

from nebuly.entities import DevelopmentPhase, Message, Publisher_T, Watched


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
        phase: DevelopmentPhase | None,
        publish: Publisher_T,
    ) -> None:
        self._api_key = api_key
        self._project = project
        self._phase = phase
        self._publisher = publish

    def on_event_received(self, watched: Watched) -> None:
        self._set_nebuly_kwargs(watched)
        self._validate_phase(watched)
        message = Message(
            api_key=self._api_key,
            watched=watched,
        )
        self._publisher(message)

    def _set_nebuly_kwargs(self, watched: Watched) -> None:
        if "nebuly_project" not in watched.called_with_nebuly_kwargs and self._project:
            watched.called_with_nebuly_kwargs["nebuly_project"] = self._project
        if "nebuly_phase" not in watched.called_with_nebuly_kwargs and self._phase:
            watched.called_with_nebuly_kwargs["nebuly_phase"] = self._phase

    @staticmethod
    def _validate_phase(watched: Watched) -> None:
        if "nebuly_phase" not in watched.called_with_nebuly_kwargs:
            raise ValueError("nebuly_phase must be set")
        if not isinstance(
            watched.called_with_nebuly_kwargs["nebuly_phase"], DevelopmentPhase
        ):
            raise ValueError("nebuly_phase must be a DevelopmentPhase")
