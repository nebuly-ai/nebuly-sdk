from typing import Callable

from anthropic import AsyncStream, Stream


def is_anthropic_generator(function: Callable) -> bool:
    return isinstance(function, (Stream, AsyncStream))
