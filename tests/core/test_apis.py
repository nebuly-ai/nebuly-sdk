import unittest
from unittest.mock import MagicMock, patch

import nebuly.core.apis as nebuly
from nebuly.core.queues import NebulyQueue
from nebuly.core.schemas import DevelopmentPhase, TagData, Task


class TestAPIs(unittest.TestCase):
    @patch("nebuly.core.apis.NebulyQueue")
    @patch("nebuly.core.apis.NebulyTrackingDataThread")
    @patch("nebuly.core.apis._instantiate_trackers")
    def test_init__is_initializing_the_queue(
        self,
        mocked_instantiate_trackers: MagicMock,
        mocked_thread: MagicMock,
        mocked_queue: MagicMock,
    ) -> None:
        with patch("nebuly.core.apis._instantiate_trackers"):
            nebuly.init(
                project="test_project",
                phase=DevelopmentPhase.EXPERIMENTATION,
            )

        self.assertIsInstance(obj=nebuly._nebuly_queue, cls=MagicMock)

    @patch("nebuly.core.apis.NebulyQueue")
    @patch("nebuly.core.apis.NebulyTrackingDataThread")
    @patch("nebuly.core.apis._instantiate_trackers")
    def test_init__is_initializing_the_thread(
        self,
        mocked_instantiate_trackers: MagicMock,
        mocked_thread: MagicMock,
        mocked_queue: MagicMock,
    ) -> None:
        import nebuly

        nebuly.api_key = "test_api_key"
        with patch("nebuly.core.apis._instantiate_trackers"):
            nebuly.init(
                project="test_project",
                phase=DevelopmentPhase.EXPERIMENTATION,
            )

        mocked_thread.assert_called_once()

    @patch("nebuly.core.apis.NebulyQueue")
    @patch("nebuly.core.apis.NebulyTrackingDataThread")
    @patch("nebuly.core.apis._instantiate_trackers")
    def test_init__is_initializing_the_trackers_list(
        self,
        mocked_instantiate_trackers: MagicMock,
        mocked_thread: MagicMock,
        mocked_queue: MagicMock,
    ) -> None:
        import nebuly

        nebuly.api_key = "test_api_key"
        nebuly.init(
            project="test_project",
            phase=DevelopmentPhase.EXPERIMENTATION,
        )

        mocked_instantiate_trackers.assert_called_once()

    @patch("nebuly.core.apis.NebulyQueue")
    @patch("nebuly.core.apis.NebulyTrackingDataThread")
    @patch("nebuly.trackers.openai.OpenAITracker")
    def test_init__is_initializing_openai_tracker(
        self,
        mocked_openai_tracker: MagicMock,
        mocked_thread: MagicMock,
        mocked_queue: MagicMock,
    ) -> None:
        import nebuly

        nebuly.api_key = "test_api_key"
        mocked_openai_tracker_instance: MagicMock = MagicMock()
        mocked_openai_tracker.return_value = mocked_openai_tracker_instance
        mocked_openai_tracker_instance.replace_sdk_functions = MagicMock()
        nebuly.init(
            project="test_project",
            phase=DevelopmentPhase.EXPERIMENTATION,
        )

        mocked_openai_tracker.assert_called_once()
        mocked_openai_tracker_instance.replace_sdk_functions.assert_called_once()

    def test_init__is_detecting_wrong_project_type(self) -> None:
        error_text = "project must be of type str"
        with self.assertRaises(expected_exception=TypeError) as context:
            nebuly.init(
                project=1,
                phase=DevelopmentPhase.EXPERIMENTATION,
            )

        self.assertTrue(expr=error_text in str(object=context.exception))

    def test_init__is_detecting_wrong_phase_type(self) -> None:
        error_text = "development_phase must be of type DevelopmentPhase"
        with self.assertRaises(expected_exception=TypeError) as context:
            nebuly.init(
                project="test_project",
                phase=1,
            )

        self.assertTrue(expr=error_text in str(object=context.exception))

    def test_init__is_detecting_wrong_task_type(self) -> None:
        error_text = "task must be of type Task"
        with self.assertRaises(expected_exception=TypeError) as context:
            nebuly.init(
                project="test_project",
                phase=DevelopmentPhase.EXPERIMENTATION,
                task=1,
            )

        self.assertTrue(expr=error_text in str(object=context.exception))

    def test_context_manager__is_replacing_the_tracker_info(
        self,
    ) -> None:
        tag_data = TagData(
            project="test_project",
            development_phase=DevelopmentPhase.EXPERIMENTATION,
            task=Task.TEXT_GENERATION,
        )
        nebuly._nebuly_queue = NebulyQueue(tag_data=tag_data)

        with nebuly.tracker(
            project="test_project_2",
            development_phase=DevelopmentPhase.PRODUCTION,
            task=Task.CHAT,
        ):
            tag_data = nebuly._nebuly_queue.tag_data
            self.assertEqual(first=tag_data.project, second="test_project_2")
            self.assertEqual(
                first=tag_data.development_phase, second=DevelopmentPhase.PRODUCTION
            )
            self.assertEqual(first=tag_data.task, second=Task.CHAT)

        tag_data: TagData = nebuly._nebuly_queue.tag_data
        self.assertEqual(first=tag_data.project, second="test_project")
        self.assertEqual(
            first=tag_data.development_phase, second=DevelopmentPhase.EXPERIMENTATION
        )
        self.assertEqual(first=tag_data.task, second=Task.TEXT_GENERATION)

    def test_context_manger__is_detecting_missing_init(self) -> None:
        error_text = "nebuly.init()"
        nebuly._nebuly_queue = None
        with self.assertRaises(expected_exception=RuntimeError) as context:
            with nebuly.tracker(
                project="test_project_2",
                development_phase=DevelopmentPhase.PRODUCTION,
                task=Task.CHAT,
            ):
                pass

        self.assertTrue(expr=error_text in str(object=context.exception))

    def test_context_manager__is_detecting_wrong_project_type(self):
        error_text = "project must be of type str"
        with self.assertRaises(expected_exception=TypeError) as context:
            with nebuly.tracker(
                project=1,
                development_phase=DevelopmentPhase.EXPERIMENTATION,
            ):
                pass

        self.assertTrue(expr=error_text in str(object=context.exception))

    def test_context_manager__is_detecting_wrong_phase_type(self):
        error_text = "development_phase must be of type DevelopmentPhase"
        with self.assertRaises(expected_exception=TypeError) as context:
            with nebuly.tracker(
                project="test_project",
                development_phase=1,
            ):
                pass

        self.assertTrue(expr=error_text in str(object=context.exception))

    def test_context_manager__is_detecting_wrong_task_type(self) -> None:
        error_text = "task must be of type Task"
        with self.assertRaises(expected_exception=TypeError) as context:
            with nebuly.tracker(
                project="test_project",
                development_phase=DevelopmentPhase.EXPERIMENTATION,
                task=1,
            ):
                pass

        self.assertTrue(expr=error_text in str(object=context.exception))
