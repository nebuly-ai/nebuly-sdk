from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable


@dataclass(frozen=True, slots=True)
class Package:
    name: str
    versions: list[str]
    to_patch: list[str]


@dataclass(frozen=True, slots=True)
class Watched:
    function: Callable
    called_at: datetime
    called_with_args: tuple
    called_with_kwargs: dict[str, Any]
    called_with_nebuly_kwargs: dict[str, Any]
    returned: Any


@dataclass(frozen=True, slots=True)
class Message:
    api_key: str
    phase: str
    project: str
    watched: Watched


Observer_T = Callable[[Watched], None]

Publisher_T = Callable[[Message], None]
