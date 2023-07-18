from typing import Dict, Union
from unittest import TestCase
from unittest.mock import MagicMock, patch

from nebuly.core.queues import (
    DataPackageConverter,
    NebulyQueue,
    QueueObject,
    RawTrackedData,
)
from nebuly.core.schemas import (
    DevelopmentPhase,
    GenericProviderAttributes,
    NebulyDataPackage,
    Provider,
    TagData,
    Task,
)


class TestNebulyQueue(TestCase):
    default_tag_data: TagData = TagData(
        project="project",
        phase=DevelopmentPhase.PRODUCTION,
        task=Task.TEXT_CLASSIFICATION,
    )

    def test_patch_tag_data__is_updating_all_the_fields(self) -> None:
        nebuly_queue = NebulyQueue(tag_data=self.default_tag_data)
        new_tag_data: TagData = TagData(
            project="new_project",
            phase=DevelopmentPhase.EXPERIMENTATION,
            task=Task.TEXT_EDITING,
        )

        nebuly_queue.patch_tag_data(tag_data=new_tag_data)

        self.assertEqual(first=nebuly_queue.tag_data, second=new_tag_data)

    def test_patch_tag_data__is_discarding_unknown_attributes(self) -> None:
        nebuly_queue = NebulyQueue(tag_data=self.default_tag_data)
        undefined_tag_data = TagData()

        nebuly_queue.patch_tag_data(tag_data=undefined_tag_data)

        self.assertEqual(first=nebuly_queue.tag_data, second=self.default_tag_data)

    def test_put__is_calling_the_super_put(self) -> None:
        with patch.object(target=NebulyQueue, attribute="put") as mocked_put:
            nebuly_queue = NebulyQueue(tag_data=self.default_tag_data)
            queue_object_mocked = MagicMock()
            nebuly_queue.put(item=queue_object_mocked)

        mocked_put.assert_called_once_with(item=queue_object_mocked)

    def test_put__is_tagging_the_queue_object(self) -> None:
        nebuly_queue = NebulyQueue(tag_data=self.default_tag_data)
        queue_object_mocked = MagicMock()

        nebuly_queue.put(item=queue_object_mocked)

        queue_object_mocked.tag.assert_called_once()


class ImplementedDatapackageConverter(DataPackageConverter):
    def get_data_package(
        self, raw_data: RawTrackedData, tag_data: TagData
    ) -> NebulyDataPackage:
        body = GenericProviderAttributes(
            project="project",
            phase=DevelopmentPhase.PRODUCTION,
            task=Task.TEXT_CLASSIFICATION,
            api_type="api_type",
            timestamp=1234567890.1234567890,
            timestamp_end=1234567890.1234567890,
        )
        return NebulyDataPackage(
            provider=Provider.OPENAI,
            body=body,
        )


class TestQueueObject(TestCase):
    mocked_request_kwargs: Dict[str, str] = {
        "url": "url",
        "method": "method",
    }
    mocked_request_response: Dict[str, Union[int, str]] = {
        "status_code": 200,
        "content": "content",
    }
    mocked_api_type = "api_type"
    mocked_timestamp = 1234567890.1234567890

    def test_tag__is_updating_all_the_fields(self) -> None:
        queue_object = QueueObject(
            data_package_converter=ImplementedDatapackageConverter(),
            raw_data=RawTrackedData(),
        )
        tag_data = TagData(
            project="project",
            phase=DevelopmentPhase.PRODUCTION,
            task=Task.TEXT_CLASSIFICATION,
        )

        queue_object.tag(tag_data=tag_data)

        self.assertEqual(first=queue_object._tag_data, second=tag_data)

    def test_tag__is_not_linking_directly_the_tag_data(self) -> None:
        queue_object = QueueObject(
            data_package_converter=ImplementedDatapackageConverter(),
            raw_data=RawTrackedData(),
        )
        tag_data = TagData(
            project="project",
            phase=DevelopmentPhase.PRODUCTION,
            task=Task.TEXT_CLASSIFICATION,
        )

        queue_object.tag(tag_data=tag_data)

        self.assertNotEqual(first=id(queue_object._tag_data), second=id(tag_data))

    def test_tag__is_raising_value_error_for_unspecified_project(self) -> None:
        queue_object = QueueObject(
            data_package_converter=ImplementedDatapackageConverter(),
            raw_data=RawTrackedData(),
        )

        tag_data = TagData(phase=DevelopmentPhase.PRODUCTION)

        with self.assertRaises(expected_exception=ValueError) as context:
            queue_object.tag(tag_data=tag_data)

        self.assertTrue(expr="Project" in str(object=context.exception))

    def test_tag__is_raising_value_error_for_unspecified_phase(self) -> None:
        queue_object = QueueObject(
            data_package_converter=ImplementedDatapackageConverter(),
            raw_data=RawTrackedData(),
        )

        tag_data = TagData(project="my_project")

        with self.assertRaises(expected_exception=ValueError) as context:
            queue_object.tag(tag_data=tag_data)

        self.assertTrue(expr="Development phase" in str(object=context.exception))
