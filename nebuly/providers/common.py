from __future__ import annotations

import logging
import re
from typing import Any

from nebuly.entities import HistoryEntry

logger = logging.getLogger(__name__)


def extract_anthropic_input_and_history(
    prompt: str | list[dict[str, Any]]
) -> tuple[str, list[HistoryEntry]]:
    if isinstance(prompt, str):
        # Ensure that the prompt is a string following pattern "\n\nHuman:...Assistant:
        prompt = prompt.strip()
        if re.match(r"\n*Human:.*Assistant:$", prompt, re.DOTALL) is None:
            return prompt, []
        try:
            # Extract human and assistant interactions using regular expression
            pattern = re.compile(
                r"Human:(.*?)\n*Assistant:(.*?)(?=\n*Human:|$)", re.DOTALL
            )
            interactions = pattern.findall(prompt)

            # Extracting the last user input
            last_user_input = interactions[-1][0].strip()

            # Create a list of tuples for the history
            history = [
                HistoryEntry(human.strip(), assistant.strip())
                for human, assistant in interactions[:-1]
            ]

            return last_user_input, history
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("Failed to extract input and history for anthropic: %s", e)
            return prompt, []
    else:
        try:
            user_messages = [
                el["content"][0]["text"] for el in prompt if el["role"] == "user"
            ]
            assistant_messages = [
                el["content"][0]["text"] for el in prompt if el["role"] == "assistant"
            ]
            history = [
                HistoryEntry(user, assistant)
                for user, assistant in zip(user_messages[:-1], assistant_messages)
            ]
            return user_messages[-1], history
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("Failed to extract input and history for anthropic: %s", e)
            return str(prompt), []
