from __future__ import annotations

from typing import Any


def get_argument(
    args: tuple[Any, ...], kwargs: dict[str, Any], arg_name: str, arg_idx: int
) -> Any:
    """
    Get the argument both when it's passed as a positional argument or as a
    keyword argument
    """
    if kwargs.get(arg_name) is not None:
        return kwargs.get(arg_name)

    if len(args) > arg_idx:
        return args[arg_idx]

    return None
