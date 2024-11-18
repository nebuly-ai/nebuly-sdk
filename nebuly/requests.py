from __future__ import annotations

import json
import logging
import os
import ssl
import urllib.request
from time import sleep
from typing import Any

from nebuly import config
from nebuly.entities import InteractionWatch
from nebuly.exceptions import InvalidNebulyKeyError

logger = logging.getLogger(__name__)


class CustomJSONEncoder(json.JSONEncoder):
    ACCEPTED_KEYS = ["_model_id", "_model", "_context", "_examples", "_message_history"]

    @staticmethod
    def _check_key(key: str) -> bool:
        """
        Reject all private keys except for ones that contain info about the model
        """
        return not key.startswith("_") or key in CustomJSONEncoder.ACCEPTED_KEYS

    @staticmethod
    def _check_value(value: Any) -> bool:
        correct_types: tuple[type[Any], ...] = (
            str,
            int,
            float,
            bool,
            list,
            tuple,
            dict,
            type(None),
        )
        has_correct_type = isinstance(value, correct_types)
        is_valid_class = False
        try:
            if hasattr(value, "__dict__"):
                json.dumps(value.__dict__)
                is_valid_class = True
        except TypeError:
            is_valid_class = any(
                (hasattr(value, key) for key in CustomJSONEncoder.ACCEPTED_KEYS)
            )
        return has_correct_type or is_valid_class

    def default(self, o: Any) -> Any:
        if isinstance(o, bytes):
            return o.decode("utf-8")
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


def post_message(
    watched: InteractionWatch, api_key: str, anonymize: bool = True
) -> None:
    message = json.dumps(
        {"body": watched, "anonymize": anonymize}, cls=CustomJSONEncoder
    )
    url = os.environ.get(
        "NEBULY_API_URL",
        "https://backend.nebuly.com/event-ingestion/api/v1/events",
    )
    if watched.api_key:
        api_key = watched.api_key
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

    ctx = ssl.create_default_context()
    ssl_verify_mode = config.get_ssl_verify_mode()
    if ssl_verify_mode is ssl.CERT_NONE:
        ctx.check_hostname = False
    ctx.verify_mode = ssl_verify_mode

    tries = 0
    while tries < 3:
        try:
            with urllib.request.urlopen(
                request,
                timeout=3,
                context=ctx,
            ) as response:  # nosec[CWE-22]
                response_body = response.read().decode("utf-8")
                logger.debug("response_body: %s", response_body)
                return response_body
        except urllib.error.HTTPError as e:
            if e.code == 401:
                raise InvalidNebulyKeyError(
                    "Invalid Nebuly API key, please provide a valid one and try again."
                ) from e
            logger.exception(
                "HTTPError when publishing the interaction: %s, "
                "retrying in 3 seconds...",
                e,
            )
            sleep(3)
            tries += 1
        except Exception as e:  # pylint: disable=broad-except
            logger.exception(
                "Error when publishing the interaction: %s, "
                "retrying in 3 seconds...",
                e,
            )
            sleep(3)
            tries += 1

    logger.error("Failed to publish the interaction, giving up.")
    return None
