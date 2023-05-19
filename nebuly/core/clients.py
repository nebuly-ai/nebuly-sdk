import requests
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
from typing import Dict

from tenacity import (
    retry,
    wait_fixed,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
)

from nebuly.core.queues import NebulyQueue
from nebuly.core.schemas import NebulyDataPackage, NebulyRequestParams
from nebuly.utils.logger import nebuly_logger


RETRY_WAIT_TIME = 1
RETRY_STOP_TIME = 10
RETRY_STOP_ATTEMPTS = 10


class NebulyClient:
    def __init__(self, api_key: str):
        self._api_key = api_key
        self._nebuly_event_ingestion_url = "http://event_ingestion/api/v1/record"

    def send_request_to_nebuly_server(
        self,
        request_data: NebulyDataPackage,
        request_params: NebulyRequestParams,
    ):
        headers = self._get_header()
        try:
            self._post_nebuly_event_ingestion_endpoint(
                headers,
                request_data,
                request_params,
            )
        except HTTPError as errh:
            nebuly_logger.error(f"Nebuly Request, Http Error: {errh}")
        except BaseException as e:
            nebuly_logger.error(f"Nebuly Request, Error: {e}")

    def _get_header(self) -> Dict:
        headers = {"authorization": f"Bearer {self._api_key}"}
        return headers

    @retry(
        wait=wait_fixed(RETRY_WAIT_TIME),
        stop=(
            stop_after_delay(RETRY_STOP_TIME) | stop_after_attempt(RETRY_STOP_ATTEMPTS)
        ),
        retry=retry_if_exception_type(
            (Timeout, RequestException, ConnectTimeout, ReadTimeout, ConnectionError)
        ),
    )
    def _post_nebuly_event_ingestion_endpoint(
        self,
        headers: Dict,
        request_data: NebulyDataPackage,
        request_params: NebulyRequestParams,
    ):
        response = requests.post(
            url=self._nebuly_event_ingestion_url,
            headers=headers,
            data=request_data.json(),
            params=request_params.json(),
        )
        response.raise_for_status()


class NebulyTrackingDataThread(Thread):
    def __init__(
        self,
        queue: NebulyQueue,
        nebuly_client: NebulyClient,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._queue = queue
        self._nebuly_client = nebuly_client
        self.thread_running = True

    def run(self):
        while self.thread_running is True or self._queue.empty() is False:
            try:
                queue_object = self._queue.get(timeout=0)
            except Empty:
                continue
            except KeyboardInterrupt:
                self.thread_running = False
                nebuly_logger.warning(
                    "Keyboard interrupt detected. "
                    "Sending all the remaining data to Nebuly server."
                )

            request_data = queue_object.as_data_package()
            request_params = queue_object.as_request_params()
            self._nebuly_client.send_request_to_nebuly_server(
                request_data, request_params
            )
            self._queue.task_done()
