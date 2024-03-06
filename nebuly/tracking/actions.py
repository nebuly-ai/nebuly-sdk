# pylint: disable=duplicate-code
import asyncio
import json
import os
from abc import ABC
from datetime import datetime, timezone
from typing import Any, Optional

from nebuly.api_key import get_api_key
from nebuly.requests import post_json_data
from nebuly.tracking.types import (
    FeedbackAction,
    FeedbackActionMetadata,
    FeedbackActionName,
)


class TrackingBase(ABC):
    """
    Base class for ABTesting and AsyncABTesting
    """

    def __init__(self, api_key: Optional[str] = None, anonymize: bool = True) -> None:
        if api_key is None:
            api_key = get_api_key()
        self._api_key = api_key
        url = (
            os.environ.get("NEBULY_API_URL")
            or "https://backend.nebuly.com/event-ingestion/api/v1/events"
        )
        self._URL = f"{url}/feedback"
        self._anonymize = anonymize

    def get_request_payload(  # pylint: disable=too-many-arguments
        self,
        user_id: str,
        action: str,
        text: Optional[str] = None,
        input: Optional[str] = None,  # pylint: disable=redefined-builtin
        output: Optional[str] = None,
    ) -> str:
        feedback_action = FeedbackAction(
            slug=FeedbackActionName(action),
            extras={"text": text} if text else None,
        )
        action_metadata = FeedbackActionMetadata(
            input=input,
            output=output,
            end_user=user_id,
            timestamp=datetime.now(timezone.utc),
            anonymize=self._anonymize,
        )
        request_data = {
            "action": feedback_action.to_dict(),
            "metadata": action_metadata.to_dict(),
        }
        payload = json.dumps(request_data)
        return payload


class TrackingSDK(TrackingBase):
    """
    Synchronous Feedback Action tracking class
    """

    def send_feedback_action(  # pylint: disable=too-many-arguments
        self,
        user_id: str,
        action: str,
        text: Optional[str] = None,
        input: Optional[str] = None,  # pylint: disable=redefined-builtin
        output: Optional[str] = None,
    ) -> Any:
        """
        Send the feedback action to Nebuly's platform
        """
        payload = self.get_request_payload(user_id, action, text, input, output)
        return post_json_data(self._URL, payload, self._api_key)


class AsyncTrackingSDK(TrackingBase):
    """
    Asynchronous Feedback Action tracking class
    """

    async def send_feedback_action(  # pylint: disable=too-many-arguments
        self,
        user_id: str,
        action: str,
        text: Optional[str] = None,
        input: Optional[str] = None,  # pylint: disable=redefined-builtin
        output: Optional[str] = None,
    ) -> Any:
        """
        Send the feedback action to Nebuly's platform
        """
        payload = self.get_request_payload(user_id, action, text, input, output)
        loop = asyncio.get_running_loop()
        raw_response = await loop.run_in_executor(
            None,
            post_json_data,
            self._URL,
            payload,
            self._api_key,
        )
        return raw_response
