from typing import Any, Callable
from functools import wraps
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class Watched:
    function: Callable
    called_at: datetime
    called_with_args: tuple
    called_with_kwargs: dict[str, Any]
    returned: Any


Observer = Callable[[Watched], None]


def patcher(observer: Observer):
    def inner(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            called_at = datetime.now(timezone.utc)
            result = f(*args, **kwargs)
            watched = Watched(
                function=f,
                called_at=called_at,
                called_with_args=args,
                called_with_kwargs=kwargs,
                returned=result,
            )
            observer(watched)
            return result

        return wrapper

    return inner
