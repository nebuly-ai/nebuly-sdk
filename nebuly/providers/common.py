from __future__ import annotations

import logging
import re

from nebuly.entities import HistoryEntry

logger = logging.getLogger(__name__)


def extract_anthropic_input_and_history(prompt: str) -> tuple[str, list[HistoryEntry]]:
    # Ensure that the prompt is a string following pattern "\n\nHuman:...Assistant:
    prompt = prompt.strip()
    if re.match(r"\n*Human:.*Assistant:$", prompt, re.DOTALL) is None:
        return prompt, []
    try:
        # Extract human and assistant interactions using regular expression
        pattern = re.compile(r"Human:(.*?)\n*Assistant:(.*?)(?=\n*Human:|$)", re.DOTALL)
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
