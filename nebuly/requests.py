from __future__ import annotations

import logging
import urllib.request

logger = logging.getLogger(__name__)


def post_json_data(url: str, json_data: str) -> str:
    request = urllib.request.Request(
        url,
        data=json_data.encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(request, timeout=3) as response:
        response_body = response.read().decode("utf-8")
        logger.debug("response_body: %s", response_body)

    return response_body
