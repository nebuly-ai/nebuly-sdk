import inspect


def transform_args_to_kwargs(func, args, kwargs):
    """
    Transform args to kwargs based on the function signature.
    """
    sig = inspect.signature(func)
    bound_args = sig.bind(*args, **kwargs)
    complete_kwargs = dict(bound_args.arguments)

    # if signature presents only **kwargs or **params i need to unpack it
    # but it may introduce some bugs or unexpected behaviour.
    # suggestions?
    if (len(complete_kwargs) == 1) and (
        isinstance(list(complete_kwargs.values())[0], dict)
    ):
        complete_kwargs = list(complete_kwargs.values())[0]

    return complete_kwargs
