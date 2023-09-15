from __future__ import annotations

import os

import pytest

from nebuly import init
from nebuly.entities import DevelopmentPhase
from nebuly.exceptions import APIKeyNotProvidedError, NebulyAlreadyInitializedError
from nebuly.init import _get_api_key


def test_cannot_init_twice() -> None:
    init(api_key="fake_key", project="", phase=DevelopmentPhase.EXPERIMENTATION)
    with pytest.raises(NebulyAlreadyInitializedError):
        init(api_key="fake_key", project="", phase=DevelopmentPhase.EXPERIMENTATION)


def test_get_api_key_not_provided() -> None:
    with pytest.raises(APIKeyNotProvidedError):
        init()


def test_api_key_from_env() -> None:
    os.environ["NEBULY_API_KEY"] = "fake_key"
    key = _get_api_key()
    assert key == "fake_key"
    del os.environ["NEBULY_API_KEY"]
