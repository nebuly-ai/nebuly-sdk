from dataclasses import dataclass
from typing import Optional

from nebuly.core.schemas import Task


@dataclass
class PromptInfo:
    text: str
    task: Task

    def is_matching(
        self,
        prompt: "PromptInfo",
        required_matching_words: int = 5,
    ) -> bool:
        """Checks if the prompt is matching with the current prompt.
        A prompt is matching if the first required_matching_words
        are the same.

        Args:
            prompt (PromptInfo): The prompt object to compare with.
            required_matching_words (int, optional): The number of words
                that should match. Defaults to 5.

        Returns:
            bool: Whether the prompt is matching or not accordingly to the
                required number of matching words.
        """
        first_prompt_words = self.text.split(sep=" ")
        second_prompt_words = prompt.text.split(sep=" ")

        if len(first_prompt_words) < required_matching_words:
            return False
        if len(second_prompt_words) < required_matching_words:
            return False

        for i in range(required_matching_words):
            if first_prompt_words[i] != second_prompt_words[i]:
                return False
        return True

    def get_common_text(self, prompt: "PromptInfo") -> str:
        """Gets the first words that are in common between the current prompt
        object and the given prompt object.

        Args:
            prompt (PromptInfo): The prompt object to compare with.

        Returns:
            str: The common text between the two prompts.
        """
        first_prompt_words = self.text.split(sep=" ")
        second_prompt_words = prompt.text.split(sep=" ")
        number_of_words = min(len(first_prompt_words), len(second_prompt_words))

        for i in range(number_of_words):
            if first_prompt_words[i] != second_prompt_words[i]:
                common_prompt = " ".join(first_prompt_words[:i])
                return common_prompt
        return " ".join(first_prompt_words[:number_of_words])


class TaskDetector:
    def __init__(self) -> None:
        # TODO: Add check for memory usage: i.e. if i am saving too many prompts
        self._prompt_list = []

        self._keywords_dict = {
            "TEXT_SUMMARIZATION": ["summarize", "summary", "summarization"],
            "TEXT_CLASSIFICATION": ["classify", "classification"],
            "CHAT": ["chat", "chatbot", "assistant"],
        }

    def detect_task_from_text(self, text: str) -> Task:
        """Detects the task from the given text.

        Args:
            text (str): The text to detect the task from.

        Returns:
            Task: The task detected from the text.
        """
        prompt_info = PromptInfo(text=text, task=Task.UNKNOWN)
        current_prompt = self._get_prompt_from_prompt_list(prompt=prompt_info)
        self._assign_prompt_task(prompt=current_prompt)
        return current_prompt.task

    def _assign_prompt_task(self, prompt: PromptInfo) -> None:
        if prompt.task != Task.UNKNOWN:
            return
        for task_string, keywords in self._keywords_dict.items():
            for keyword in keywords:
                if keyword in prompt.text:
                    prompt.task = Task[task_string]
        if prompt.task == Task.UNKNOWN:
            prompt.task = Task.TEXT_GENERATION

    def _get_prompt_from_prompt_list(self, prompt: PromptInfo) -> PromptInfo:
        matching_prompt = self._get_matching_prompt_from_prompt_list(prompt=prompt)
        if matching_prompt is not None:
            common_prompt = matching_prompt.get_common_text(prompt=prompt)
            matching_prompt.text = common_prompt
            return matching_prompt
        else:
            self._prompt_list.append(prompt)
            return prompt

    def _get_matching_prompt_from_prompt_list(
        self, prompt: PromptInfo
    ) -> Optional[PromptInfo]:
        for p in self._prompt_list:
            if p.is_matching(prompt=prompt):
                return p
        return None
