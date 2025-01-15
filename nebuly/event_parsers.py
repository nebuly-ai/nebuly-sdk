import logging
import re
from typing import Sequence

from nebuly.entities import HistoryEntry, InteractionWatch, SpanWatch

logger = logging.getLogger(__name__)

MAX_URL_LENGTH = 2083


def remove_base64_media_from_spans(spans: Sequence[SpanWatch]) -> None:
    for span in spans:
        if span.media is None:
            continue
        no_base64_media = []
        for media in span.media:
            if not is_base64_text(media) and len(media) < MAX_URL_LENGTH:
                no_base64_media.append(media)
            else:
                logger.info(
                    "Media removed from span because it is base64 or too long: %s",
                    media[:100],
                )
        span.media = no_base64_media


def is_base64_text(text: str) -> bool:
    regex = re.compile(r"data:.*;base64,([a-zA-Z0-9+/=]+)")
    return bool(regex.match(text))


def trim_all_base64_from_interaction_schema(interaction: InteractionWatch) -> None:
    interaction.input = remove_base64(interaction.input)
    interaction.output = remove_base64(interaction.output)
    new_history = []
    for history in interaction.history:
        new_history.append(
            HistoryEntry(
                user=remove_base64(history.user),
                assistant=remove_base64(history.assistant),
            )
        )
    interaction.history = new_history


def remove_base64(text: str) -> str:
    """
    Removes long words that are likely to be base64 strings
    from a given text leaving only the first and last 10 characters.
    """
    # Regular expression to find base64 strings
    base64_chars = r"[a-zA-Z0-9+/=]"
    base64_pattern = f"{base64_chars}{{50,}}"
    base64_re = re.compile(base64_pattern)

    # Replace base64 strings with their first and last 10 characters
    def replace(match: re.Match[str]) -> str:
        return match.group()[:10] + "..." + match.group()[-10:]

    return base64_re.sub(replace, text)
