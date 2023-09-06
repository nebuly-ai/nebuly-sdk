from typing import Any, Callable
from functools import wraps
from dataclasses import dataclass
from datetime import datetime, timezone
from copy import deepcopy


@dataclass(frozen=True, slots=True)
class Watched:
    function: Callable
    called_at: datetime
    called_with_args: tuple
    called_with_kwargs: dict[str, Any]
    called_with_nebuly_kwargs: dict[str, Any]
    returned: Any


Observer_T = Callable[[Watched], None]


def split_nebuly_kwargs(
    kwargs: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    nebuly_kwargs = {}
    function_kwargs = {}
    for key in kwargs:
        if key.startswith("nebuly_"):
            nebuly_kwargs[key] = kwargs[key]
        else:
            function_kwargs[key] = kwargs[key]
    return nebuly_kwargs, function_kwargs


def patcher(observer: Observer_T):
    """
    Decorator that calls observer with a Watched instance when the decorated
    function is called
    """

    def inner(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            nebuly_kwargs, function_kwargs = split_nebuly_kwargs(kwargs)

            original_args = deepcopy(args)
            nebuly_kwargs = deepcopy(nebuly_kwargs)
            original_kwargs = deepcopy(function_kwargs)

            called_at = datetime.now(timezone.utc)

            result = f(*args, **function_kwargs)

            original_result = deepcopy(result)
            watched = Watched(
                function=f,
                called_at=called_at,
                called_with_args=original_args,
                called_with_kwargs=original_kwargs,
                called_with_nebuly_kwargs=nebuly_kwargs,
                returned=original_result,
            )
            observer(watched)

            return result

        return wrapper

    return inner
