from unittest import TestCase
from unittest.mock import patch, MagicMock

from nebuly.core.schemas import TagData
from nebuly.core.queues import NebulyQueue, QueueObject


class TestNebulyQueue(TestCase):
    def test_patch_tag_data__is_updating_all_the_fields(self):
        nebuly_queue = NebulyQueue()
        mocked_tag_data = MagicMock()
        mocked_tag_data.project = "project"
        mocked_tag_data.phase = "phase"
        mocked_tag_data.task = "task"

        nebuly_queue.patch_tag_data(mocked_tag_data)

        self.assertEqual(nebuly_queue.tag_data.project, "project")
        self.assertEqual(nebuly_queue.tag_data.phase, "phase")
        self.assertEqual(nebuly_queue.tag_data.task, "task")

    def test_patch_tag_data__is_discarding_none_objects(self):
        nebuly_queue = NebulyQueue()
        mocked_tag_data = MagicMock()
        mocked_tag_data.project = None
        mocked_tag_data.phase = None
        mocked_tag_data.task = None

        nebuly_queue.patch_tag_data(mocked_tag_data)

        self.assertEqual(nebuly_queue.tag_data.project, "unknown_project")
        self.assertEqual(nebuly_queue.tag_data.phase.value, "unknown")
        self.assertEqual(nebuly_queue.tag_data.task.value, "undetected")

    def test_put__is_calling_the_super_put(self):
        with patch.object(NebulyQueue, "put") as mocked_put:
            nebuly_queue = NebulyQueue()
            queue_object_mocked = MagicMock()
            nebuly_queue.put(queue_object_mocked)

        mocked_put.assert_called_once_with(queue_object_mocked)

    def test_put__is_tagging_the_queue_object(self):
        nebuly_queue = NebulyQueue()
        queue_object_mocked = MagicMock()
        queue_object_mocked.tag.return_value = MagicMock()
        nebuly_queue.put(queue_object_mocked)

        queue_object_mocked.tag.assert_called_once()


class TestQueueObject(TestCase):
    mocked_request_kwargs = {
        "url": "url",
        "method": "method",
    }
    mocked_request_response = {
        "status_code": 200,
        "content": "content",
    }
    mocked_api_type = "api_type"
    mocked_timestamp = 1234567890.1234567890

    def test_tag__is_updating_all_the_fields(self):
        data_package_converter_mocked = MagicMock()
        queue_object = QueueObject(
            data_package_converter=data_package_converter_mocked,
            request_kwargs=self.mocked_request_kwargs,
            request_response=self.mocked_request_response,
            api_type=self.mocked_api_type,
            timestamp=self.mocked_timestamp,
        )

        tag_data = TagData(project="project", phase="phase", task="task")

        queue_object.tag(tag_data)

        self.assertEqual(queue_object._tag_data.project, "project")
        self.assertEqual(queue_object._tag_data.phase, "phase")
        self.assertEqual(queue_object._tag_data.task, "task")

    def test_tag__is_raising_value_error_for_none_project(self):
        data_package_converter_mocked = MagicMock()
        queue_object = QueueObject(
            data_package_converter=data_package_converter_mocked,
            request_kwargs=self.mocked_request_kwargs,
            request_response=self.mocked_request_response,
            api_type=self.mocked_api_type,
            timestamp=self.mocked_timestamp,
        )

        tag_data = TagData(project=None, phase="phase", task="task")

        with self.assertRaises(ValueError) as context:
            queue_object.tag(tag_data)

        self.assertTrue("Project" in str(context.exception))

    def test_tag__is_raising_value_error_for_none_phase(self):
        data_package_converter_mocked = MagicMock()
        queue_object = QueueObject(
            data_package_converter=data_package_converter_mocked,
            request_kwargs=self.mocked_request_kwargs,
            request_response=self.mocked_request_response,
            api_type=self.mocked_api_type,
            timestamp=self.mocked_timestamp,
        )

        tag_data = TagData(project="project", phase=None, task="task")

        with self.assertRaises(ValueError) as context:
            queue_object.tag(tag_data)

        print(str(context.exception))
        self.assertTrue("Development phase" in str(context.exception))

    def test_as_data_package__is_setting_data_into_the_data_package(self):
        data_package_converter_mocked = MagicMock()
        data_package_converter_mocked.set_tag_data.return_value = MagicMock()
        data_package_converter_mocked.get_data_package.return_value = MagicMock()
        queue_object = QueueObject(
            data_package_converter=data_package_converter_mocked,
            request_kwargs=self.mocked_request_kwargs,
            request_response=self.mocked_request_response,
            api_type=self.mocked_api_type,
            timestamp=self.mocked_timestamp,
        )
        tag_data = TagData(project="project", phase="phase", task="task")
        queue_object.tag(tag_data)
        queue_object.as_data_package()

        data_package_converter_mocked.get_data_package.assert_called_once_with(
            tag_data=tag_data,
            request_kwargs=self.mocked_request_kwargs,
            request_response=self.mocked_request_response,
            api_type=self.mocked_api_type,
            timestamp=self.mocked_timestamp,
        )
