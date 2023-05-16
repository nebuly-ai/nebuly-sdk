import inspect
from typing import Dict, Optional, Tuple

import mutagen


def transform_args_to_kwargs(
    func: callable, func_args: Tuple, func_kwargs: Dict
) -> Dict:
    """Transforms the function args and kwargs to only a kwargs dict.

    Args:
        func (callable): The function whose args and kwargs should be transformed.
        func_args (Tuple): The function args.
        func_kwargs (Dict): The function kwargs.

    Returns:
        Dict: The kwargs dict.
    """
    sig = inspect.signature(func)
    bound_args = sig.bind(*func_args, **func_kwargs)
    complete_kwargs = dict(bound_args.arguments)

    # if signature presents only **kwargs or **params i need to unpack it
    # but it may introduce some bugs or unexpected behaviour.
    # suggestions?
    if (len(complete_kwargs) == 1) and (
        isinstance(list(complete_kwargs.values())[0], dict)
    ):
        complete_kwargs = list(complete_kwargs.values())[0]

    return complete_kwargs


def get_media_file_length_in_seconds(file_path: str) -> Optional[int]:
    """Gets the length of the media file in seconds.

    Args:
        file_path (str): The path to the media file.

    Returns:
        Optional[int]: The length of the media file in seconds.
    """
    try:
        audio_file = mutagen.File(file_path)
    except Exception:
        return None
    return int(audio_file.info.length)
