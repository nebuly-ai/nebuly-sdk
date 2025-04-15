import json

import pytest

from nebuly.entities import InteractionWatch
from nebuly.requests import _crop_spans


def test_crop_spans__is_working() -> None:
    with open("./tests/huge_message.json", encoding="utf-8") as f:
        message = json.load(f)

    old_size = len(str(message).encode("utf-8")) / 1_000_000
    assert old_size > 0.5, "Initial size should be bigger"

    new_message = json.loads(_crop_spans(json.dumps(message)))

    new_size = len(str(new_message).encode("utf-8")) / 1_000_000
    assert new_size < 0.5, "New size is not smaller"

    fields_to_check = ["input", "output", "time_start", "time_end", "end_user"]
    for field in fields_to_check:
        new_field = new_message["body"][field]
        old_field = message["body"][field]
        assert new_field == old_field, "Message content has changed"

    try:
        _ = InteractionWatch(**message["body"])
    except (TypeError, ValueError, KeyError) as e:
        pytest.fail(f"Creating InteractionWatch raised an error: {e}")


def test_crop_spans__is_doing_nothing_if_size_is_small() -> None:
    with open("./tests/huge_message.json", encoding="utf-8") as f:
        message = json.load(f)

    old_size = len(str(message).encode("utf-8")) / 1_000_000
    assert old_size < 1, "Initial size should be lower"

    new_message = json.loads(_crop_spans(json.dumps(message), max_size=1))

    new_size = len(str(new_message).encode("utf-8")) / 1_000_000
    assert new_size == old_size, "New size should be identical to the original"

    try:
        _ = InteractionWatch(**message["body"])
    except (TypeError, ValueError, KeyError) as e:
        pytest.fail(f"Creating InteractionWatch raised an error: {e}")
