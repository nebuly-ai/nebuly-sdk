from typing import Any

from google.generativeai.types import text_types

from nebuly.providers.utils import get_argument


def extract_google_input_and_history(
    original_args: tuple[Any],
    original_kwargs: dict[str, Any],
    function_name: str,
) -> tuple[str, list[tuple[str, Any]]]:
    if function_name == "generativeai.generate_text":
        return original_kwargs.get("prompt"), []
    if function_name in ["generativeai.chat", "generativeai.chat_async"]:
        history = [
            ("user" if i % 2 == 0 else "assistant", el)
            for i, el in enumerate(original_kwargs.get("messages")[:-1])
            if len(original_kwargs.get("messages")) > 1
        ]
        return original_kwargs.get("messages")[-1], history
    if function_name == "generativeai.discuss.ChatResponse.reply":
        prompt = get_argument(original_args, original_kwargs, "message", 1)
        history = [
            ("user" if el["author"] == "0" else "assistant", el["content"])
            for el in getattr(original_args[0], "messages", [])
        ]
        return prompt, history


def extract_google_output(function_name: str, output: text_types.Completion) -> str:
    if function_name == "generativeai.generate_text":
        return output.result
    if function_name in [
        "generativeai.chat",
        "generativeai.chat_async",
        "generativeai.discuss.ChatResponse.reply",
    ]:
        return output.messages[-1]["content"]
