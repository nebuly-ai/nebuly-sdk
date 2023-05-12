import unittest

from nebuly.core.schemas import Task
from nebuly.utils.task_detector import TaskDetector


class TestPromptDiscriminator(unittest.TestCase):
    def test_discriminate__is_adding_prompts_when_prompt_list_is_empty(
        self,
    ):
        task_detector = TaskDetector()
        task_detector.detect_task_from_text("I am a test prompt and I am here to stay")

        self.assertEqual(
            task_detector._prompt_list[0].text,
            "I am a test prompt and I am here to stay",
        )

    def test_discriminate__is_adding_prompts_when_prompt_list_is_not_empty(
        self,
    ):
        task_detector = TaskDetector()
        task_detector.detect_task_from_text("I am a test prompt and I am here to stay")
        task_detector.detect_task_from_text(
            "I am another test prompt and I am different"
        )

        self.assertEqual(
            task_detector._prompt_list[0].text,
            "I am a test prompt and I am here to stay",
        )
        self.assertEqual(
            task_detector._prompt_list[1].text,
            "I am another test prompt and I am different",
        )
        self.assertEqual(len(task_detector._prompt_list), 2)

    def test_discriminate__is_detecting_identical_prompts_as_one(self):
        task_detector = TaskDetector()
        task_detector.detect_task_from_text("I am a test prompt and I am here to stay")
        task_detector.detect_task_from_text("I am a test prompt and I am here to stay")

        self.assertEqual(
            task_detector._prompt_list[0].text,
            "I am a test prompt and I am here to stay",
        )
        self.assertEqual(len(task_detector._prompt_list), 1)

    def test_discriminate__is_detecting_only_the_common_text(self):
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

    def test_discriminate__is_detecting_the_task_from_keywords(self):
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
