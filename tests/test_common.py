from __future__ import annotations

from nebuly.entities import HistoryEntry
from nebuly.providers.common import extract_anthropic_input_and_history


def test_extract_anthropic_input_and_history__wrong_format_1() -> None:
    prompt = "\n\nHello"
    assert extract_anthropic_input_and_history(prompt) == (prompt.strip(), [])


def test_extract_anthropic_input_and_history__wrong_format_2() -> None:
    prompt = "\n\nHuman:say hi"
    assert extract_anthropic_input_and_history(prompt) == (prompt.strip(), [])


def test_extract_anthropic_input_and_history__wrong_format_3() -> None:
    prompt = "\n\nAssistant:say hi"
    assert extract_anthropic_input_and_history(prompt) == (prompt.strip(), [])


def test_extract_anthropic_input_and_history__wrong_format_4() -> None:
    prompt = "\n\nHuman:say hi\n\nAssistant:hi\n\nHuman:how are you?"
    assert extract_anthropic_input_and_history(prompt) == (prompt.strip(), [])


def test_extract_anthropic_input_and_history__no_history() -> None:
    prompt = "\n\nHuman:say hi\n\nAssistant:"
    assert extract_anthropic_input_and_history(prompt) == ("say hi", [])


def test_extract_anthropic_input_and_history__with_history() -> None:
    prompt = "\n\nHuman:say hi\n\nAssistant:hi\n\nHuman:how are you?\n\nAssistant:"
    assert extract_anthropic_input_and_history(prompt) == (
        "how are you?",
        [
            HistoryEntry(user="say hi", assistant="hi"),
        ],
    )


def test_extract_anthropic_input_and_history__no_return_chars() -> None:
    prompt = "Human:say hi Assistant:hi Human:how are you? Assistant:"
    assert extract_anthropic_input_and_history(prompt) == (
        "how are you?",
        [
            HistoryEntry(user="say hi", assistant="hi"),
        ],
    )


def test_extract_anthropic_input_and_history__no_return_chars__no_history() -> None:
    prompt = "Human:say hi Assistant:"
    assert extract_anthropic_input_and_history(prompt) == ("say hi", [])


def test_extract_anthropic_input_and_history__return_chars_in_text__case_1() -> None:
    prompt = (
        "\n\nHuman:say hi\n\nAssistant:hi\n\nHuman:how \n\nare you?\n\nAssistant:"
        "I'm \n\n good\n\nHuman:ok\n\nAssistant:"
    )
    assert extract_anthropic_input_and_history(prompt) == (
        "ok",
        [
            HistoryEntry(user="say hi", assistant="hi"),
            HistoryEntry(user="how \n\nare you?", assistant="I'm \n\n good"),
        ],
    )


def test_extract_anthropic_input_and_history__return_chars_in_text__case_2() -> None:
    prompt = (
        "Human:say hiAssistant:hiHuman:how \n\nare you?Assistant:I'm \n\n good"
        "\n\nHuman:okAssistant:"
    )
    assert extract_anthropic_input_and_history(prompt) == (
        "ok",
        [
            HistoryEntry(user="say hi", assistant="hi"),
            HistoryEntry(user="how \n\nare you?", assistant="I'm \n\n good"),
        ],
    )
