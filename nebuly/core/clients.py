import logging
from http import HTTPStatus
from queue import Empty
from threading import Thread
from typing import Dict, Any

import requests
from requests import (
    ConnectionError,
    ConnectTimeout,
    HTTPError,
    ReadTimeout,
    Timeout,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
    wait_fixed,
)

from nebuly.core.queues import NebulyQueue
from nebuly.core.schemas import NebulyDataPackage

nebuly_logger = logging.getLogger(name=__name__)
nebuly_logger.setLevel(level=logging.INFO)

RETRY_WAIT_TIME = 1
RETRY_STOP_TIME = 10
RETRY_STOP_ATTEMPTS = 10


class RetryHTTPException(Exception):
    def __init__(self, status_code: int, error_message: str):
        self.status_code = status_code
        self.error_message = error_message
        super().__init__(f"{status_code} {error_message}")


class NebulyClient:
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._nebuly_event_ingestion_url = "https://event_ingestion/api/v1/record"

    def send_request_to_nebuly_server(
        self,
        request_data: NebulyDataPackage,
    ) -> None:
        """Send request to Nebuly server.

        Args:
            request_data (NebulyDataPackage): The request data.
        """
        headers = {"Authorization": f"Bearer {self._api_key}"}
        try:
            self._post_nebuly_event_ingestion_endpoint(
                headers=headers,
                request_data=request_data,
            )
        except HTTPError as errh:
            nebuly_logger.error(
                msg=f"An error occurred while communicating with the Nebuly Server.\n"
                f"HTTP Error: {errh}\n",
                exc_info=True,
            )
        except BaseException as e:
            # Here is possible to add more specific exceptions,
            # but even importing all the exception imported in the base request module
            # there are still some exceptions that are not being caught.
            # since the user-code should never fail for our fault,
            # here I am catching all the exceptions and logging them.
            nebuly_logger.error(
                msg=f"An error occurred while communicating with the Nebuly Server.\n"
                f"Generic Error: {e}\n",
                exc_info=True,
            )

    @retry(
        wait=wait_fixed(wait=RETRY_WAIT_TIME),
        stop=(
            stop_after_delay(max_delay=RETRY_STOP_TIME)
            | stop_after_attempt(max_attempt_number=RETRY_STOP_ATTEMPTS)
        ),
        retry=retry_if_exception_type(
            exception_types=(
                Timeout,
                ConnectTimeout,
                RetryHTTPException,
                ReadTimeout,
                ConnectionError,
            )
        ),
    )
    def _post_nebuly_event_ingestion_endpoint(
        self,
        headers: Dict[str, str],
        request_data: NebulyDataPackage,
    ) -> None:
        response = None
        try:
            response = requests.post(
                url=self._nebuly_event_ingestion_url,
                headers=headers,
                data=request_data.json(),
            )
            response.raise_for_status()
        except HTTPError as e:
            if response and response.status_code == HTTPStatus.SERVICE_UNAVAILABLE:
                raise RetryHTTPException(HTTPStatus.SERVICE_UNAVAILABLE, str(e)) from e
            elif response and response.status_code == HTTPStatus.GATEWAY_TIMEOUT:
                raise RetryHTTPException(HTTPStatus.GATEWAY_TIMEOUT, str(e)) from e
            else:
                raise e


class NebulyTrackingDataThread(Thread):
    def __init__(
        self,
        queue: NebulyQueue,
        nebuly_client: NebulyClient,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(**kwargs)

        self._queue = queue
        self._nebuly_client = nebuly_client
        self.thread_running = True
        self.force_exit = False

    def run(self) -> None:
        """Continuously takes elements from the queue and sends them to the
        Nebuly server.
        """
        while self.thread_running is True or self._queue.empty() is False:
            if self.force_exit is True:
                break

            try:
                queue_object = self._queue.get(timeout=0)
            except Empty:
                continue

            request_data = queue_object.as_data_package()
            self._nebuly_client.send_request_to_nebuly_server(request_data=request_data)
            self._queue.task_done()
