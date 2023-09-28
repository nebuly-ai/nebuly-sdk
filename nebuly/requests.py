from __future__ import annotations

import json
import logging
import os
import urllib.request
from typing import Any
from unittest.mock import Mock

from nebuly.entities import InteractionWatch

logger = logging.getLogger(__name__)


class CustomJSONEncoder(json.JSONEncoder):
    @staticmethod
    def _check_key(key: str):
        """
        Reject all private keys except for ones that contain info about the model
        """
        return not key.startswith("_") or "model" in key

    @staticmethod
    def _check_value(value: Any):
        correct_types = str | int | float | bool | list | tuple | dict | None
        has_correct_type = isinstance(value, correct_types)
        is_valid_class = not isinstance(value, Mock) and hasattr(
            value, "_model_id"
        )  # Needed for vertexai chat models  # noqa: E501
        return has_correct_type or is_valid_class

    def default(self, o: Any) -> Any:
        if hasattr(o, "to_dict"):
            return o.to_dict()
        try:
            return json.JSONEncoder.default(self, o)
        except Exception:  # pylint: disable=broad-except
            if hasattr(o, "__dict__"):
                dict_repr = o.__dict__
                return {
                    key: dict_repr[key]
                    for key in dict_repr
                    if self._check_key(key) and self._check_value(dict_repr[key])
                }
            return str(o)


def post_message(watched: InteractionWatch, api_key: str) -> None:
    message = json.dumps({"body": watched}, cls=CustomJSONEncoder)
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
