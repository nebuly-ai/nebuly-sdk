from unittest import TestCase
from unittest.mock import patch

from nebuly.utils.functions import (
    transform_args_to_kwargs,
    get_media_file_length_in_seconds,
)


class TestFunctions(TestCase):
    def test_transform_args_to_kwargs__is_transforming_args_to_kwargs(self):
        def test_function(a, b, c, d=1, e=2):
            pass

        args = [1, 2, 3]
        kwargs = {"d": 4, "e": 5}
        complete_kwargs = transform_args_to_kwargs(test_function, args, kwargs)

        self.assertEqual(complete_kwargs, {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5})

    def test_transform_args_to_kwargs__is_not_returning_nested_dict(self):
        def test_function(*args, **kwargs):
            pass

        args = []
        kwargs = {"a": 1}
        complete_kwargs = transform_args_to_kwargs(test_function, args, kwargs)

        self.assertEqual(complete_kwargs, {"a": 1})

    @patch("nebuly.utils.functions.mutagen.File")
    def test_get_media_file_length_in_seconds__is_returning_audio_length(
        self, mocked_mutagen_file
    ):
        mocked_mutagen_file.return_value.info.length = 10.0
        self.assertEqual(get_media_file_length_in_seconds("test"), 10)
