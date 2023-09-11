from __future__ import annotations

import json
import logging
import os
import urllib.request

from nebuly.entities import Watched

logger = logging.getLogger(__name__)


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if hasattr(o, "to_dict"):
            return o.to_dict()
        try:
            return json.JSONEncoder.default(self, o)
        except Exception:  # pylint: disable=broad-except
            return str(o)


def post_message(message: Watched, api_key: str) -> None:
    json_data = json.dumps({"body": message}, cls=CustomJSONEncoder)
    url = os.environ.get(
        "NEBULY_API_URL", "https://backend.nebuly.com/event-ingestion/api/v1/events"
    )
    post_json_data(url, json_data, api_key)


def post_json_data(url: str, json_data: str, api_key: str) -> str:
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
