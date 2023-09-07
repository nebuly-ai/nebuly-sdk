import os

import pytest

from nebuly import init
from nebuly.exceptions import APIKeyNotProvidedError, NebulyAlreadyInitializedError
from nebuly.init import _get_api_key


def test_cannot_init_twice():
    init(api_key="fake_key", project="", phase="")
    with pytest.raises(NebulyAlreadyInitializedError):
        init(api_key="fake_key", project="", phase="")


def test_get_api_key_not_provided():
    with pytest.raises(APIKeyNotProvidedError):
        init()


def test_api_key_from_env():
    os.environ["NEBULY_API_KEY"] = "fake_key"
    key = _get_api_key()
    assert key == "fake_key"
    del os.environ["NEBULY_API_KEY"]
