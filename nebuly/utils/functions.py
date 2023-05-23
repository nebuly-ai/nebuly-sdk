import inspect
from datetime import datetime
from inspect import Signature, BoundArguments
from typing import Dict, Optional, Tuple, Any

import mutagen


def transform_args_to_kwargs(
    func: Any,
    func_args: Tuple[Any],
    func_kwargs: Dict[str, Any],
    specific_keyword: Optional[str] = None,
) -> Dict[str, Any]:
    """Transforms the function args and kwargs to only a kwargs dict.

    Args:
        func (Callable): The function whose args and kwargs should be transformed.
        func_args (Tuple): The function args.
        func_kwargs (Dict): The function kwargs.
        specific_keyword (str): The specific keyword to be used as a key additionally
            to kwargs.

    Returns:
        Dict: The kwargs dict.
    """
    sig: Signature = inspect.signature(obj=func)
    bound_args: BoundArguments = sig.bind(*func_args, **func_kwargs)
    complete_kwargs = dict(bound_args.arguments)

    # if there is the kwargs dictonary in the args, we need to merge it with the
    # complete kwargs dict
    if "kwargs" in complete_kwargs:
        complete_kwargs = {**complete_kwargs, **complete_kwargs["kwargs"]}
        del complete_kwargs["kwargs"]

    if specific_keyword is not None:
        if specific_keyword in complete_kwargs:
            complete_kwargs = {**complete_kwargs, **complete_kwargs[specific_keyword]}
            del complete_kwargs[specific_keyword]

    return complete_kwargs


def get_media_file_length_in_seconds(file_path: str) -> Optional[int]:
    """Gets the length of the media file in seconds.

    Args:
        file_path (str): The path to the media file.

    Returns:
        Optional[int]: The length of the media file in seconds.
    """
    try:
        audio_file: Any = mutagen.File(filething=file_path)  # type: ignore
    except Exception:
        return None
    return int(audio_file.info.length)


def get_current_timestamp() -> float:
    """Gets the current timestamp.

    Returns:
        float: The current timestamp using UTC as timezone.
    """
    return datetime.utcnow().timestamp()
