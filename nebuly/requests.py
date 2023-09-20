from __future__ import annotations

import json
import logging
import os
import urllib.request
from typing import Any

from nebuly.entities import ChainEvent, Watched

logger = logging.getLogger(__name__)


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if hasattr(o, "to_dict"):
            return o.to_dict()
        try:
            return json.JSONEncoder.default(self, o)
        except Exception:  # pylint: disable=broad-except
            return str(o)


def post_message(watched: Watched | ChainEvent, api_key: str) -> None:
    message = json.dumps({"body": watched, "provider": ""}, cls=CustomJSONEncoder)
    url = os.environ.get(
        # TODO: make this url configurable
        "NEBULY_API_URL",
        "https://backend.nebuly.com/event-ingestion/api/v1/events",
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

    with urllib.request.urlopen(request, timeout=3) as response:  # nosec
        response_body = response.read().decode("utf-8")
        logger.debug("response_body: %s", response_body)

    return response_body
