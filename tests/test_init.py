import pytest

from nebuly import init
from nebuly.exceptions import NebulyAlreadyInitializedError


def test_cannot_init_twice():
    init(api_key="", project="", phase="")
    with pytest.raises(NebulyAlreadyInitializedError):
        init(api_key="", project="", phase="")
