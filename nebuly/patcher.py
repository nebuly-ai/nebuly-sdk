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
    returned: Any


Observer_T = Callable[[Watched], None]


def patcher(observer: Observer_T):
    """
    Decorator that calls observer with a Watched instance when the decorated
    function is called
    """

    def inner(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            original_args = deepcopy(args)
            original_kwargs = deepcopy(kwargs)
            called_at = datetime.now(timezone.utc)
            result = f(*args, **kwargs)
            original_result = deepcopy(result)
            watched = Watched(
                function=f,
                called_at=called_at,
                called_with_args=original_args,
                called_with_kwargs=original_kwargs,
                returned=original_result,
            )
            observer(watched)
            return result

        return wrapper

    return inner
