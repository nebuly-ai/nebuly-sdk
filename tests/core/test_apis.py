import unittest
from unittest.mock import MagicMock, patch

import nebuly.core.apis as nebuly

from nebuly.core.schemas import DevelopmentPhase, Task, TagData
from nebuly.core.clients import NebulyQueue


class TestAPIs(unittest.TestCase):
    @patch("nebuly.core.apis.NebulyQueue")
    @patch("nebuly.core.apis.NebulyTrackingDataThread")
    @patch("nebuly.core.apis._instantiate_trackers")
    def test_init__is_initializing_the_queue(
        self, mocked_instantiate_trackers, mocked_thread, mocked_queue
    ):
        with patch("nebuly.core.apis._instantiate_trackers"):
            nebuly.init(
                project="test_project",
                phase=DevelopmentPhase.DEVELOPMENT,
            )

        self.assertIsInstance(nebuly._nebuly_queue, MagicMock)

    @patch("nebuly.core.apis.NebulyQueue")
    @patch("nebuly.core.apis.NebulyTrackingDataThread")
    @patch("nebuly.core.apis._instantiate_trackers")
    def test_init__is_initializing_the_thread(
        self, mocked_instantiate_trackers, mocked_thread, mocked_queue
    ):
        with patch("nebuly.core.apis._instantiate_trackers"):
            nebuly.init(
                project="test_project",
                phase=DevelopmentPhase.DEVELOPMENT,
            )

        mocked_thread.assert_called_once()

    @patch("nebuly.core.apis.NebulyQueue")
    @patch("nebuly.core.apis.NebulyTrackingDataThread")
    @patch("nebuly.core.apis._instantiate_trackers")
    def test_init__is_initializing_the_trackers_list(
        self, mocked_instantiate_trackers, mocked_thread, mocked_queue
    ):
        nebuly.init(
            project="test_project",
            phase=DevelopmentPhase.DEVELOPMENT,
        )

        mocked_instantiate_trackers.assert_called_once()

    @patch("nebuly.core.apis.NebulyQueue")
    @patch("nebuly.core.apis.NebulyTrackingDataThread")
    @patch("nebuly.trackers.openai.OpenAITracker")
    def test_init__is_initializing_openai_tracker(
        self,
        mocked_openai_tracker,
        mocked_thread,
        mocked_queue,
    ):
        mocked_openai_tracker_instance = MagicMock()
        mocked_openai_tracker.return_value = mocked_openai_tracker_instance
        mocked_openai_tracker_instance.replace_sdk_functions = MagicMock()
        nebuly.init(
            project="test_project",
            phase=DevelopmentPhase.DEVELOPMENT,
        )

        mocked_openai_tracker.assert_called_once()
        mocked_openai_tracker_instance.replace_sdk_functions.assert_called_once()

    def test_init__is_detecting_wrong_project_type(self):
        error_text = "project must be of type str"
        with self.assertRaises(TypeError) as context:
            nebuly.init(
                project=1,
                phase=DevelopmentPhase.DEVELOPMENT,
            )

        self.assertTrue(error_text in str(context.exception))

    def test_init__is_detecting_wrong_phase_type(self):
        error_text = "phase must be of type DevelopmentPhase"
        with self.assertRaises(TypeError) as context:
            nebuly.init(
                project="test_project",
                phase=1,
            )

        self.assertTrue(error_text in str(context.exception))

    def test_init__is_detecting_wrong_task_type(self):
        error_text = "task must be of type Task"
        with self.assertRaises(TypeError) as context:
            nebuly.init(
                project="test_project",
                phase=DevelopmentPhase.DEVELOPMENT,
                task=1,
            )

        self.assertTrue(error_text in str(context.exception))

    def test_context_manager__is_replacing_the_tracker_info(
        self,
    ):
        nebuly._nebuly_queue = NebulyQueue()
        tagged_data = TagData(
            project="test_project",
            phase=DevelopmentPhase.DEVELOPMENT,
            task=Task.TEXT_GENERATION,
        )
        nebuly._nebuly_queue.tagged_data = tagged_data

        with nebuly.tracker(
            project="test_project_2",
            phase=DevelopmentPhase.PRODUCTION,
            task=Task.CHAT,
        ):
            tagged_data = nebuly._nebuly_queue.tagged_data
            self.assertEqual(tagged_data.project, "test_project_2")
            self.assertEqual(tagged_data.phase, DevelopmentPhase.PRODUCTION)
            self.assertEqual(tagged_data.task, Task.CHAT)

        tagged_data = nebuly._nebuly_queue.tagged_data
        self.assertEqual(tagged_data.project, "test_project")
        self.assertEqual(tagged_data.phase, DevelopmentPhase.DEVELOPMENT)
        self.assertEqual(tagged_data.task, Task.TEXT_GENERATION)

    def test_context_manger__is_detecting_missing_init(self):
        error_text = "nebuly.init()"
        nebuly._nebuly_queue = None
        with self.assertRaises(RuntimeError) as context:
            with nebuly.tracker(
                project="test_project_2",
                phase=DevelopmentPhase.PRODUCTION,
                task=Task.CHAT,
            ):
                pass

        self.assertTrue(error_text in str(context.exception))

    def test_context_manager__is_detecting_wrong_project_type(self):
        error_text = "project must be of type str"
        with self.assertRaises(TypeError) as context:
            with nebuly.tracker(
                project=1,
                phase=DevelopmentPhase.DEVELOPMENT,
            ):
                pass

        self.assertTrue(error_text in str(context.exception))

    def test_context_manager__is_detecting_wrong_phase_type(self):
        error_text = "phase must be of type DevelopmentPhase"
        with self.assertRaises(TypeError) as context:
            with nebuly.tracker(
                project="test_project",
                phase=1,
            ):
                pass

        self.assertTrue(error_text in str(context.exception))

    def test_context_manager__is_detecting_wrong_task_type(self):
        error_text = "task must be of type Task"
        with self.assertRaises(TypeError) as context:
            with nebuly.tracker(
                project="test_project",
                phase=DevelopmentPhase.DEVELOPMENT,
                task=1,
            ):
                pass

        self.assertTrue(error_text in str(context.exception))
