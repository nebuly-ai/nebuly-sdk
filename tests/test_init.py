import pytest

from nebuly.exceptions import AlreadyImportedError
from nebuly.init import init


def test_check_openai_already_imported():
    import openai  # noqa: F401

    with pytest.raises(AlreadyImportedError):
        init()
