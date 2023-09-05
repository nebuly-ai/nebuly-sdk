from functools import wraps


def patcher(observer):
    def inner(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            result = f(*args, **kwargs)
            observer((f, args, kwargs, result))
            return result

        return wrapper

    return inner
