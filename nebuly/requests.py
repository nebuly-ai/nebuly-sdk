from __future__ import annotations

import logging
import os
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)


def post_message(message: str, api_key: str) -> None:
    url = os.environ.get(
        "NEBULY_API_URL", "https://backend.nebuly.com/event-ingestion/api/v1/events"
    )
    post_json_data(url, message, api_key)


def post_json_data(url: str, json_data: str, api_key: str) -> Any:
    request = urllib.request.Request(
        url,
        data=json_data.encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )

    with urllib.request.urlopen(request, timeout=3) as response:
        response_body = response.read().decode("utf-8")
        logger.debug("response_body: %s", response_body)

    return response_body
