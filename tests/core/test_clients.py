from unittest import TestCase
from unittest.mock import MagicMock, patch
import time

from nebuly.core.clients import NebulyTrackingDataThread, NebulyClient
from nebuly.core.schemas import (
    NebulyDataPackage,
    NebulyRequestParams,
    DevelopmentPhase,
    Task,
    Provider,
)


class TestNebulyClient(TestCase):
    mocked_request_params = NebulyRequestParams(kind=Provider.OPENAI)
    mocked_data_package = NebulyDataPackage(
        model="test-model",
        project="test-project",
        phase=DevelopmentPhase.EXPERIMENTATION,
        task=Task.AUDIO_TRANSCRIPTION,
        timestamp=1234567890,
        api_type="test-api-type",
        n_prompt_tokens=1,
        n_output_tokens=2,
        n_output_images=3,
        image_size="test-image-size",
        audio_duration_seconds=4,
        training_file_id="test-training-file-id",
        training_id="test-training-id",
    )

    @patch("nebuly.core.clients.requests")
    def test_send_request_to_nebuly_server__is_issuing_post_request(
        self, mocked_requests
    ):
        mocked_requests.post = MagicMock()
        nebuly_client = NebulyClient(api_key="test-key")

        nebuly_client.send_request_to_nebuly_server(
            self.mocked_data_package, self.mocked_request_params
        )

        mocked_headers = {"authorization": "Bearer test-key"}

        mocked_requests.post.assert_called_once_with(
            url=nebuly_client._nebuly_event_ingestion_url,
            headers=mocked_headers,
            data=self.mocked_data_package.json(),
            params=self.mocked_request_params.json(),
        )


class TestNebulyTrackingDataThread(TestCase):
    def test_run__is_consuming_objects_from_the_queue(self):
        mocked_nebuly_queue = MagicMock()
        mocked_nebuly_queue.get = MagicMock()
        mocked_nebuly_queue.task_done = MagicMock()
        mocked_nebuly_client = MagicMock()

        nebuly_tracking_data_thread = NebulyTrackingDataThread(
            mocked_nebuly_queue,
            nebuly_client=mocked_nebuly_client,
        )
        nebuly_tracking_data_thread.start()
        time.sleep(0.001)
        nebuly_tracking_data_thread.thread_running = False
        nebuly_tracking_data_thread.join()

        mocked_nebuly_queue.get.assert_called()
        mocked_nebuly_queue.task_done.assert_called()

    def test_run__is_sending_request_to_nebuly_server(self):
        mocked_nebuly_queue = MagicMock()
        mocked_nebuly_queue.get = MagicMock()
        mocked_nebuly_client = MagicMock()
        mocked_nebuly_client.send_request_to_nebuly_server = MagicMock()

        nebuly_tracking_data_thread = NebulyTrackingDataThread(
            mocked_nebuly_queue,
            nebuly_client=mocked_nebuly_client,
        )
        nebuly_tracking_data_thread.start()
        time.sleep(0.001)
        nebuly_tracking_data_thread.thread_running = False
        nebuly_tracking_data_thread.join()

        mocked_nebuly_client.send_request_to_nebuly_server.assert_called()

    def test_run__is_issuing_the_creation_of_the_request_data(self):
        mocked_nebuly_queue = MagicMock()
        mocked_object = MagicMock()
        mocked_object.as_data_package = MagicMock()
        mocked_nebuly_queue.get = MagicMock(return_value=mocked_object)
        mocked_nebuly_client = MagicMock()

        nebuly_tracking_data_thread = NebulyTrackingDataThread(
            mocked_nebuly_queue,
            nebuly_client=mocked_nebuly_client,
        )
        nebuly_tracking_data_thread.start()
        time.sleep(0.001)
        nebuly_tracking_data_thread.thread_running = False
        nebuly_tracking_data_thread.join()

        mocked_object.as_data_package.assert_called()
