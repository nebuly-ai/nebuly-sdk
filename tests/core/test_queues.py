from unittest import TestCase
from unittest.mock import patch, MagicMock

from nebuly.core.schemas import TagData
from nebuly.core.queues import NebulyQueue, QueueObject, DataPackageConverter


class TestNebulyQueue(TestCase):
    def test_patch_tagged_data__is_updating_all_the_fields(self):
        nebuly_queue = NebulyQueue()
        tagged_data_mocked = MagicMock()
        tagged_data_mocked.project = "project"
        tagged_data_mocked.phase = "phase"
        tagged_data_mocked.task = "task"

        nebuly_queue.patch_tagged_data(tagged_data_mocked)

        self.assertEqual(nebuly_queue.tagged_data.project, "project")
        self.assertEqual(nebuly_queue.tagged_data.phase, "phase")
        self.assertEqual(nebuly_queue.tagged_data.task, "task")

    def test_patch_tagged_data__is_discarding_none_objects(self):
        nebuly_queue = NebulyQueue()
        tagged_data_mocked = MagicMock()
        tagged_data_mocked.project = None
        tagged_data_mocked.phase = None
        tagged_data_mocked.task = None

        nebuly_queue.patch_tagged_data(tagged_data_mocked)

        self.assertEqual(nebuly_queue.tagged_data.project, "unknown_project")
        self.assertEqual(nebuly_queue.tagged_data.phase.value, "unknown")
        self.assertEqual(nebuly_queue.tagged_data.task.value, "undetected")

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
    def test_tag__is_updating_all_the_fields(self):
        queue_object = QueueObject(
            request_kwargs=MagicMock(),
            request_response=MagicMock(),
            api_type=MagicMock(),
            data_package_converter=MagicMock(),
        )

        tagged_data = TagData(project="project", phase="phase", task="task")

        queue_object.tag(tagged_data)

        self.assertEqual(queue_object._tagged_data.project, "project")
        self.assertEqual(queue_object._tagged_data.phase, "phase")
        self.assertEqual(queue_object._tagged_data.task, "task")

    def test_tag__is_raising_value_error_for_none_project(self):
        queue_object = QueueObject(
            request_kwargs=MagicMock(),
            request_response=MagicMock(),
            api_type=MagicMock(),
            data_package_converter=MagicMock(),
        )

        tagged_data = TagData(project=None, phase="phase", task="task")

        with self.assertRaises(ValueError) as context:
            queue_object.tag(tagged_data)

        self.assertTrue("Project" in str(context.exception))

    def test_tag__is_raising_value_error_for_none_phase(self):
        queue_object = QueueObject(
            request_kwargs=MagicMock(),
            request_response=MagicMock(),
            api_type=MagicMock(),
            data_package_converter=MagicMock(),
        )

        tagged_data = TagData(project="project", phase=None, task="task")

        with self.assertRaises(ValueError) as context:
            queue_object.tag(tagged_data)

        print(str(context.exception))
        self.assertTrue("Development phase" in str(context.exception))

    def test_as_data_package__is_loading_data_into_the_data_package(self):
        data_package_converter_mocked = MagicMock()
        data_package_converter_mocked.load_tag_data.return_value = MagicMock()
        data_package_converter_mocked.load_request_data.return_value = MagicMock()
        data_package_converter_mocked.get_data_package.return_value = MagicMock()
        queue_object = QueueObject(
            request_kwargs=MagicMock(),
            request_response=MagicMock(),
            api_type=MagicMock(),
            data_package_converter=data_package_converter_mocked,
        )

        queue_object._tagged_data = TagData(
            project="project", phase="phase", task="task"
        )

        queue_object.as_data_package()

        data_package_converter_mocked.load_tag_data.assert_called_once()
        data_package_converter_mocked.load_request_data.assert_called_once()
        data_package_converter_mocked.get_data_package.assert_called_once()


class ImplementedDataPackageConverter(DataPackageConverter):
    def get_data_package(self):
        pass


class TestDataPackageConverter(TestCase):
    def test_load_tag_data__is_assigning_tagged_data(self):
        data_package_converter = ImplementedDataPackageConverter()
        tagged_data = TagData(project="project", phase="phase", task="task")
        data_package_converter.load_tag_data(tagged_data)

        self.assertEqual(data_package_converter._tagged_data.project, "project")
        self.assertEqual(data_package_converter._tagged_data.phase, "phase")
        self.assertEqual(data_package_converter._tagged_data.task, "task")
        self.assertIsNot(data_package_converter._tagged_data, tagged_data)

    def test_load_request_data__is_assigning_request_data(self):
        data_package_converter = ImplementedDataPackageConverter()
        request_kwargs = {"key": "value"}
        request_response = {"key": "value"}
        api_type = "api_type"
        timestamp = 10202020
        data_package_converter.load_request_data(
            request_kwargs, request_response, api_type, timestamp
        )

        self.assertEqual(data_package_converter._request_kwargs, request_kwargs)
        self.assertEqual(data_package_converter._request_response, request_response)
        self.assertEqual(data_package_converter._api_type, api_type)
        self.assertEqual(data_package_converter._timestamp, timestamp)
