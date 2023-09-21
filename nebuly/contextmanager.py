import inspect
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Generator
from uuid import UUID

from nebuly.entities import InteractionWatch, Observer, SpanWatch


class InteractionContextError(Exception):
    pass


class InteractionContextInitiationError(InteractionContextError):
    pass


class AlreadyInInteractionContext(InteractionContextError):
    pass


class NotInInteractionContext(InteractionContextError):
    pass


class InteractionMustBeLocalVariable(InteractionContextError):
    pass


class InteractionContext:
    def __init__(
        self,
        user: str | None = None,
        user_group_profile: str | None = None,
        input: str | None = None,
        output: str | None = None,
        history: list[tuple[str, str]] = None,
        hierarchy: dict[UUID, UUID] = None,
        spans: list[SpanWatch] = None,
        do_not_call_directly: bool = False,
    ) -> None:
        if not do_not_call_directly:
            raise InteractionContextInitiationError(
                "Interaction cannot be directly instantiate, use the"
                " 'new_interaction' contextmanager"
            )
        self.user = user
        self.user_group_profile = user_group_profile
        self.input = input
        self.output = output
        self.spans = [] if spans is None else spans
        self.history = {} if history is None else history
        self.hierarchy = {} if hierarchy is None else hierarchy
        self.time_start = datetime.now(timezone.utc)
        self.observer = None
        self.finished = False

    def set_input(self, value: str) -> None:
        self.input = value

    def set_history(self, value: list[tuple[str, str]]) -> None:
        self.history = value

    def set_output(self, value: str) -> None:
        self.output = value

    def set_observer(self, observer: Observer) -> None:
        if self.observer is None:
            self.observer = observer

    def add_span(self, value: SpanWatch) -> None:
        self.spans.append(value)

    def finish(self) -> None:
        self.finished = True
        self.observer(self.as_interaction_watch())

    def set_user(self, value: str) -> None:
        self.user = value

    def set_user_group_profile(self, value: str) -> None:
        self.user_group_profile = value

    def as_interaction_watch(self) -> InteractionWatch:
        return InteractionWatch(
            input=self.input,
            output=self.output,
            time_start=self.time_start,
            time_end=datetime.now(timezone.utc),
            spans=self.spans,
            history=self.history,
            hierarchy=self.hierarchy,
            end_user=self.user,
            end_user_group_profile=self.user_group_profile,
        )


def get_nearest_open_interaction() -> InteractionContext:
    frames = inspect.stack()
    for frame in frames[::-1]:
        for v in frame.frame.f_locals.values():
            if isinstance(v, InteractionContext) and not v.finished:
                return v
    raise NotInInteractionContext()


@contextmanager
def new_interaction(
    user: str | None = None, group_profile: str | None = None
) -> Generator[InteractionContext, None, None]:
    try:
        get_nearest_open_interaction()
        raise AlreadyInInteractionContext()
    except NotInInteractionContext:
        yield InteractionContext(user, group_profile, do_not_call_directly=True)
        try:
            interaction = get_nearest_open_interaction()
        except NotInInteractionContext:
            raise InteractionMustBeLocalVariable()  # pylint: disable=raise-missing-from
        interaction.finish()
