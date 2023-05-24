from queue import Empty
from requests import (
    ConnectionError,
    ConnectTimeout,
    HTTPError,
    ReadTimeout,
    RequestException,
    Timeout,
)
from threading import Thread
from typing import Dict, Any
import requests

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
    wait_fixed,
)

from nebuly.core.queues import NebulyQueue, QueueObject
from nebuly.core.schemas import NebulyDataPackage
from nebuly.utils.logger import nebuly_logger


RETRY_WAIT_TIME = 1
RETRY_STOP_TIME = 10
RETRY_STOP_ATTEMPTS = 10


class NebulyClient:
    def __init__(self, api_key: str) -> None:
        self._api_key: str = api_key
        self._nebuly_event_ingestion_url = "https://event_ingestion/api/v1/record"

    def send_request_to_nebuly_server(
        self,
        request_data: NebulyDataPackage,
    ) -> None:
        """Send request to Nebuly server.

        Args:
            request_data (NebulyDataPackage): The request data.
        """
        headers: Dict[str, str] = self._get_header()
        try:
            self._post_nebuly_event_ingestion_endpoint(
                headers=headers,
                request_data=request_data,
            )
        except HTTPError as errh:
            nebuly_logger.error(msg=f"Nebuly Request, Http Error: {errh}")
        except BaseException as e:
            # Here is possible to add more specific exceptions,
            # but even importing all the exception imported in the base request module
            # there are still some exceptions that are not being caught.
            # since the user-code should never fail for our fault,
            # here i am catching all the exceptions and logging them.
            nebuly_logger.error(msg=f"Nebuly Request, Error: {e}")

    def _get_header(self) -> Dict[str, str]:
        headers: Dict[str, str] = {"authorization": f"Bearer {self._api_key}"}
        return headers

    @retry(
        wait=wait_fixed(wait=RETRY_WAIT_TIME),
        stop=(
            stop_after_delay(max_delay=RETRY_STOP_TIME)
            | stop_after_attempt(max_attempt_number=RETRY_STOP_ATTEMPTS)
        ),
        retry=retry_if_exception_type(
            exception_types=(
                Timeout,
                RequestException,
                ConnectTimeout,
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
        response: Any = requests.post(
            url=self._nebuly_event_ingestion_url,
            headers=headers,
            data=request_data.json(),
        )
        response.raise_for_status()


class NebulyTrackingDataThread(Thread):
    def __init__(
        self,
        queue: NebulyQueue,
        nebuly_client: NebulyClient,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(**kwargs)

        self._queue: NebulyQueue = queue
        self._nebuly_client: NebulyClient = nebuly_client
        self.thread_running = True
        self.force_exit = False

    def run(self) -> None:
        """Thread.run() that sends the data to Nebuly server.
        It gets the data from the queue, uses the queue.as_data_package() method
        to convert the queue object to a data package and then uses the nebuly_client
        to send the data to the Nebuly server.
        """
        while self.thread_running is True or self._queue.empty() is False:
            if self.force_exit is True:
                break

            try:
                queue_object: QueueObject = self._queue.get(timeout=0)
            except Empty:
                continue

            request_data: NebulyDataPackage = queue_object.as_data_package()
            self._nebuly_client.send_request_to_nebuly_server(request_data=request_data)
            self._queue.task_done()
