from queue import Empty
import requests
from threading import Thread
from typing import Dict
from urllib3.exceptions import (
    TimeoutError,
)

from nebuly.core.queues import NebulyQueue
from nebuly.core.schemas import NebulyDataPackage, NebulyRequestParams
from nebuly.utils.logger import nebuly_logger

from tenacity import (
    retry,
    wait_fixed,
    retry_if_exception,
    stop_after_attempt,
    stop_after_delay,
)


RETRY_WAIT_TIME = 1
RETRY_STOP_TIME = 10
RETRY_STOP_ATTEMPTS = 10


class NebulyClient:
    def __init__(self, api_key: str):
        self._api_key = api_key
        self._nebuly_event_ingestion_url = "http://event_ingestion/api/v1/record"

    @retry(
        wait=wait_fixed(RETRY_WAIT_TIME),
        retry=retry_if_exception(
            (
                requests.exceptions.Timeout,
                requests.exceptions.RequestException,
                TimeoutError,
            )
        ),
        stop=(
            stop_after_delay(RETRY_STOP_TIME) | stop_after_attempt(RETRY_STOP_ATTEMPTS)
        ),
    )
    def send_request_to_nebuly_server(
        self,
        request_data: NebulyDataPackage,
        request_params: NebulyRequestParams,
    ):
        headers = self._get_header()
        self._post_nebuly_event_ingestion_endpoint(
            headers,
            request_data,
            request_params,
        )

    def _get_header(self) -> Dict:
        headers = {"authorization": f"Bearer {self._api_key}"}
        return headers

    def _post_nebuly_event_ingestion_endpoint(
        self,
        headers: Dict,
        request_data: NebulyDataPackage,
        request_params: NebulyRequestParams,
    ):
        try:
            response = requests.post(
                url=self._nebuly_event_ingestion_url,
                headers=headers,
                data=request_data.json(),
                params=request_params.json(),
            )
            response.raise_for_status()
        except requests.exceptions.HTTPError as errh:
            nebuly_logger.error(f"Http Error: {errh}")
        except requests.exceptions.ConnectionError as errc:
            nebuly_logger.error(f"Error Connecting: {errc}")
        except requests.exceptions.Timeout as errt:
            nebuly_logger.error(f"Timeout Error: {errt}")
        except requests.exceptions.RequestException as err:
            nebuly_logger.error(f"Error Performing the request: {err}")


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
        while self.thread_running or self._queue.empty() is False:
            try:
                queue_object = self._queue.get()
            except Empty:
                continue
            except KeyboardInterrupt:
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
