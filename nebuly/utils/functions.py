import inspect
from typing import Dict, Optional, Tuple

import mutagen


def transform_args_to_kwargs(
    func: callable, func_args: Tuple, func_kwargs: Dict
) -> Dict:
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
    try:
        audio_file = mutagen.File(file_path)
    except Exception:
        return None
    return int(audio_file.info.length)
