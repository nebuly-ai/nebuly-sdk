from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable


@dataclass(frozen=True, slots=True)
class Package:
    """
    Package represents a package to be patched.
    """

    name: str
    versions: list[str]
    to_patch: list[str]


@dataclass(frozen=True, slots=True)
class Watched:  # pylint: disable=too-many-instance-attributes
    """
    Watched represents a call to a function that was patched.
    """

    function: Callable
    called_start: datetime
    called_end: datetime
    called_with_args: tuple
    called_with_kwargs: dict[str, Any]
    called_with_nebuly_kwargs: dict[str, Any]
    returned: Any
    generator: bool
    generator_first_element_timestamp: datetime | None


@dataclass(frozen=True, slots=True)
class Message:
    """
    Message represents a message to be sent to the API.
    """

    api_key: str
    phase: str
    project: str
    watched: Watched


Observer_T = Callable[[Watched], None]

Publisher_T = Callable[[Message], None]
