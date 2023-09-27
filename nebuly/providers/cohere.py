from typing import Callable

from cohere.responses.chat import StreamingChat
from cohere.responses.generation import StreamingGenerations


def is_cohere_generator(function: Callable) -> bool:
    return isinstance(function, (StreamingGenerations, StreamingChat))
