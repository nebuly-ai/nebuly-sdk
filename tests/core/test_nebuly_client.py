from unittest import TestCase
from unittest.mock import MagicMock, patch
import time

from nebuly.core.nebuly_client import NebulyQueue, NebulyTrackingDataThread


class TestNebulyQueue(TestCase):
    def test_update_tagged_data__is_updating_all_the_fields(self):
        nebuly_queue = NebulyQueue()
        tagged_data_mocked = MagicMock()
        tagged_data_mocked.project = "project"
        tagged_data_mocked.phase = "phase"
        tagged_data_mocked.task = "task"

        nebuly_queue.update_tagged_data(tagged_data_mocked)

        self.assertEqual(nebuly_queue.tagged_data.project, "project")
        self.assertEqual(nebuly_queue.tagged_data.phase, "phase")
        self.assertEqual(nebuly_queue.tagged_data.task, "task")

    def test_update_tagged_data__is_discarding_none_objects(self):
        nebuly_queue = NebulyQueue()
        tagged_data_mocked = MagicMock()
        tagged_data_mocked.project = None
        tagged_data_mocked.phase = None
        tagged_data_mocked.task = None

        nebuly_queue.update_tagged_data(tagged_data_mocked)

        self.assertEqual(nebuly_queue.tagged_data.project, "unknown_project")
        self.assertEqual(nebuly_queue.tagged_data.phase.value, "unknown")
        self.assertEqual(nebuly_queue.tagged_data.task.value, "undetected")

    def test_put__is_calling_the_super_put(self):
        with patch.object(NebulyQueue, "put") as mocked_put:
            nebuly_queue = NebulyQueue()
            queue_object_mocked = MagicMock()
            nebuly_queue.put(queue_object_mocked)

        mocked_put.assert_called_once_with(queue_object_mocked)

    def test_put__is_assigning_the_tagged_data_to_the_queue_object(self):
        nebuly_queue = NebulyQueue()
        queue_object_mocked = MagicMock()
        nebuly_queue.put(queue_object_mocked)

        self.assertEqual(queue_object_mocked._project, "unknown_project")
        self.assertEqual(queue_object_mocked._phase.value, "unknown")
        self.assertEqual(queue_object_mocked._task.value, "undetected")


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
        mocked_object.get_request_data = MagicMock()
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

        mocked_object.get_request_data.assert_called()
