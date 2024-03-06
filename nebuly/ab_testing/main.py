# pylint: disable=duplicate-code
import asyncio
import json
import os
from abc import ABC
from typing import Any, Optional, Sequence

from nebuly.ab_testing.types import Request, Response
from nebuly.api_key import get_api_key
from nebuly.requests import post_json_data


class ABTestingBase(ABC):
    """
    Base class for ABTesting and AsyncABTesting
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        if api_key is None:
            api_key = get_api_key()
        self._api_key = api_key
        url = os.environ.get(
            "NEBULY_AB_TESTING_API_URL",
            "https://backend.nebuly.com/api/external/variants",
        )
        self._URL = url

    @staticmethod
    def parse_raw_response(raw_response: Any) -> Response:
        """
        Parse the raw response into a Response object
        """
        json_response = json.loads(raw_response)
        reponse = Response.from_dict(json_response)
        return reponse

    @staticmethod
    def get_request_payload(user: str, feature_flags: Sequence[str]) -> str:
        request_data: Request = {
            "user": user,
            "feature_flags": feature_flags,
        }
        payload = json.dumps(request_data)
        return payload


class ABTesting(ABTestingBase):
    """
    Synchronous ABTesting class
    """

    def get_variants(self, user: str, feature_flags: Sequence[str]) -> Response:
        """
        Get the variant for each feature flag for a given user
        """
        payload = self.get_request_payload(user, feature_flags)
        raw_response = post_json_data(self._URL, payload, self._api_key)
        return self.parse_raw_response(raw_response)


class AsyncABTesting(ABTestingBase):
    """
    Asynchronous ABTesting class
    """

    async def get_variants(self, user: str, feature_flags: Sequence[str]) -> Response:
        """
        Get the variant for each feature flag for a given user
        """
        payload = self.get_request_payload(user, feature_flags)
        loop = asyncio.get_running_loop()
        raw_response = await loop.run_in_executor(
            None,
            post_json_data,
            self._URL,
            payload,
            self._api_key,
        )
        return self.parse_raw_response(raw_response)
