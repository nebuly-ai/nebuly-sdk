import json
import logging
import urllib.request

logger = logging.getLogger(__name__)


def post_json_data(url: str, data_dict: dict[str, str]):
    json_data = json.dumps(data_dict, default=str).encode("utf-8")

    logger.debug("json_data: %s", json_data)
    request = urllib.request.Request(
        url, data=json_data, headers={"Content-Type": "application/json"}
    )

    with urllib.request.urlopen(request, timeout=3) as response:
        response_body = response.read().decode("utf-8")
        logger.debug("response_body: %s", response_body)

    return response_body
