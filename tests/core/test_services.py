import unittest

from nebuly.core.schemas import Task
from nebuly.core.services import TaskDetector


class TestTaskDetector(unittest.TestCase):
    mocked_prompt1 = "I am a test prompt and I am here to stay"
    mocked_prompt2 = "I am another test prompt and I am different"

    def test_detect_task_from_text__is_adding_prompts_when_prompt_list_is_empty(
        self,
    ):
        task_detector = TaskDetector()
        task_detector.detect_task_from_text(self.mocked_prompt1)

        self.assertEqual(
            task_detector._prompt_list[0].text,
            self.mocked_prompt1,
        )

    def test_detect_task_from_text__is_adding_prompts_when_prompt_list_is_not_empty(
        self,
    ):
        task_detector = TaskDetector()
        task_detector.detect_task_from_text(self.mocked_prompt1)
        task_detector.detect_task_from_text(self.mocked_prompt2)

        self.assertEqual(
            task_detector._prompt_list[0].text,
            self.mocked_prompt1,
        )
        self.assertEqual(
            task_detector._prompt_list[1].text,
            self.mocked_prompt2,
        )
        self.assertEqual(len(task_detector._prompt_list), 2)

    def test_detect_task_from_text__is_detecting_identical_prompts_as_one(self):
        task_detector = TaskDetector()
        task_detector.detect_task_from_text(self.mocked_prompt1)
        task_detector.detect_task_from_text(self.mocked_prompt1)

        self.assertEqual(
            task_detector._prompt_list[0].text,
            self.mocked_prompt1,
        )
        self.assertEqual(len(task_detector._prompt_list), 1)

    def test_detect_task_from_text__is_detecting_only_the_common_text(self):
        task_detector = TaskDetector()
        task_detector.detect_task_from_text(
            (
                "write the summary of the following text: "
                "The quick brown fox jumps over the lazy dog"
            )
        )
        task_detector.detect_task_from_text(
            (
                "write the summary of the following text: "
                "A car was stolen from the parking lot"
            )
        )

        self.assertEqual(
            "write the summary of the following text:",
            task_detector._prompt_list[0].text,
        )
        self.assertEqual(len(task_detector._prompt_list), 1)

    def test_detect_task_from_text__is_detecting_the_task_from_keywords(self):
        prompt_detector = TaskDetector()
        task = prompt_detector.detect_task_from_text(
            (
                "write the summary of the following text: "
                "A car was stolen from the parking lot"
            )
        )
        self.assertEqual(
            Task.TEXT_SUMMARIZATION.value,
            task.value,
        )
