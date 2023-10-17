from __future__ import annotations

import logging
import re

from nebuly.entities import HistoryEntry

logger = logging.getLogger(__name__)


def extract_anthropic_input_and_history(prompt: str) -> tuple[str, list[HistoryEntry]]:
    try:
        # Extract human and assistant interactions using regular expression
        interactions = re.findall(r"Human:(.+)\n*Assistant:(.+)\n*", prompt)

        # Extract the last user input
        user_inputs = re.findall(r"Human:(.+)\n*Assistant:", prompt)
        last_user_input = user_inputs[-1].strip()

        # Create a list of tuples for the history
        history = [
            HistoryEntry(human.strip(), assistant.strip())
            for human, assistant in interactions
        ]

        return last_user_input, history
    except Exception as e:  # pylint: disable=broad-except
        logger.warning("Failed to extract input and history for anthropic: %s", e)
        return prompt, []
