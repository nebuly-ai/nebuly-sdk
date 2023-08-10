import time
from typing import Dict
from unittest import TestCase
from unittest.mock import MagicMock, patch

from nebuly.core.clients import NebulyClient
from nebuly.core.queues import NebulyTrackingDataThread
from nebuly.core.schemas import (
    DevelopmentPhase,
    GenericProviderAttributes,
    NebulyDataPackage,
    Provider,
    Task,
)


class TestNebulyClient(TestCase):
    mocked_data_package = NebulyDataPackage(
        provider=Provider.OPENAI,
        body=GenericProviderAttributes(
            project="test_project",
            development_phase=DevelopmentPhase.EXPERIMENTATION,
            task=Task.UNKNOWN,
            timestamp=1620000000,
            timestamp_end=1620000001,
        ),
    )

    @patch("nebuly.core.clients.requests")
    def test_send_request_to_nebuly_server__is_issuing_post_request(
        self,
        mocked_requests: MagicMock,
    ) -> None:
        mocked_requests.post = MagicMock()
        nebuly_client = NebulyClient(api_key="test-key")

        nebuly_client.send_request_to_nebuly_server(
            request_data=self.mocked_data_package,
        )

        mocked_headers: Dict[str, str] = {"Authorization": "Bearer test-key"}

        mocked_requests.post.assert_called_once_with(
            url=nebuly_client._nebuly_event_ingestion_url,
            headers=mocked_headers,
            data=self.mocked_data_package.json(),
        )


class TestNebulyTrackingDataThread(TestCase):
    def test_run__is_consuming_objects_from_the_queue(self) -> None:
        mocked_nebuly_queue = MagicMock()
        mocked_nebuly_queue.get = MagicMock()
        mocked_nebuly_queue.task_done = MagicMock()
        mocked_nebuly_client = MagicMock()

        nebuly_tracking_data_thread = NebulyTrackingDataThread(
            queue=mocked_nebuly_queue,
            nebuly_client=mocked_nebuly_client,
        )
        nebuly_tracking_data_thread.start()
        time.sleep(0.001)
        nebuly_tracking_data_thread.thread_running = False
        nebuly_tracking_data_thread.join()

        mocked_nebuly_queue.get.assert_called()
        mocked_nebuly_queue.task_done.assert_called()

    def test_run__is_sending_request_to_nebuly_server(self) -> None:
        mocked_nebuly_queue = MagicMock()
        mocked_nebuly_queue.get = MagicMock()
        mocked_nebuly_client = MagicMock()
        mocked_nebuly_client.send_request_to_nebuly_server = MagicMock()

        nebuly_tracking_data_thread = NebulyTrackingDataThread(
            queue=mocked_nebuly_queue,
            nebuly_client=mocked_nebuly_client,
        )
        nebuly_tracking_data_thread.start()
        time.sleep(0.001)
        nebuly_tracking_data_thread.thread_running = False
        nebuly_tracking_data_thread.join()

        mocked_nebuly_client.send_request_to_nebuly_server.assert_called()

    def test_run__is_issuing_the_creation_of_the_request_data(self) -> None:
        mocked_nebuly_queue = MagicMock()
        mocked_object = MagicMock()
        mocked_object.as_data_package = MagicMock()
        mocked_nebuly_queue.get = MagicMock(return_value=mocked_object)
        mocked_nebuly_client = MagicMock()

        nebuly_tracking_data_thread = NebulyTrackingDataThread(
            queue=mocked_nebuly_queue,
            nebuly_client=mocked_nebuly_client,
        )
        nebuly_tracking_data_thread.start()
        time.sleep(0.001)
        nebuly_tracking_data_thread.thread_running = False
        nebuly_tracking_data_thread.join()

        mocked_object.as_data_package.assert_called()
