from nebuly.patcher import Watched, patcher
from hypothesis import strategies as st
from hypothesis import given


any = st.one_of(
    st.integers(), st.floats(), st.text(), st.booleans(), st.none(), st.binary()
)


class Observer:
    def __init__(self):
        self.watched = []

    def __call__(self, watched: Watched):
        self.watched.append(watched)


@given(args=st.tuples(any), kwargs=st.dictionaries(st.text(), any))
def test_patcher_doesnt_change_any_behavior(args, kwargs):
    def to_patched(*args: int, **kwargs: str):
        """This is the docstring to be tested"""
        return args, kwargs

    patched = patcher(lambda _: None)(to_patched)

    assert patched(*args, **kwargs) == to_patched(*args, **kwargs)
    assert patched.__name__ == to_patched.__name__
    assert patched.__doc__ == to_patched.__doc__
    assert patched.__module__ == to_patched.__module__
    assert patched.__qualname__ == to_patched.__qualname__
    assert patched.__annotations__ == to_patched.__annotations__


@given(args=st.tuples(any), kwargs=st.dictionaries(st.text(), any))
def test_patcher_calls_observer(args, kwargs):
    def to_patched(*args: int, **kwargs: str):
        """This is the docstring to be tested"""
        return args, kwargs

    observer = Observer()

    patched = patcher(observer)(to_patched)

    patched(*args, **kwargs)

    assert len(observer.watched) == 1
    watched = observer.watched[0]
    assert watched.function == to_patched
    assert watched.called_with_args == args
    assert watched.called_with_kwargs == kwargs
    assert watched.returned == to_patched(*args, **kwargs)


def test_watched_is_immutable():
    def to_patched(mutable: list):
        mutable.append(1)
        return mutable

    observer = Observer()
    mutable = []

    patcher(observer)(to_patched)(mutable)

    mutable.append(2)

    assert len(observer.watched) == 1
    watched = observer.watched[0]
    assert watched.called_with_args == ([],)
    assert watched.returned == [1]
