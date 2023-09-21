import inspect
from contextlib import contextmanager
from typing import Generator


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
    def __init__(self, user: str, do_not_call_directly: bool = False) -> None:
        if not do_not_call_directly:
            raise InteractionContextInitiationError(
                "Interaction cannot be directly instantiate, use the"
                " 'new_interaction' contextmanager"
            )
        self.user = user
        self.finished = False

    def finish(self) -> None:
        self.finished = True


def get_nearest_open_interaction() -> InteractionContext:
    frames = inspect.stack()
    for frame in frames[::-1]:
        for v in frame.frame.f_locals.values():
            if isinstance(v, InteractionContext) and not v.finished:
                return v
    raise NotInInteractionContext()


@contextmanager
def new_interaction(user: str) -> Generator[InteractionContext, None, None]:
    try:
        get_nearest_open_interaction()
        raise AlreadyInInteractionContext()
    except NotInInteractionContext:
        yield InteractionContext(user, do_not_call_directly=True)
        try:
            interaction = get_nearest_open_interaction()
        except NotInInteractionContext:
            raise InteractionMustBeLocalVariable()  # pylint: disable=raise-missing-from
        interaction.finish()
