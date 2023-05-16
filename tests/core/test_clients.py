from unittest import TestCase
from unittest.mock import MagicMock
import time

from nebuly.core.clients import NebulyTrackingDataThread


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
