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
        first_prompt_words = self.text.split(" ")
        second_prompt_words = prompt.text.split(" ")

        if len(first_prompt_words) < required_matching_words:
            return False
        if len(second_prompt_words) < required_matching_words:
            return False

        for i in range(required_matching_words):
            if first_prompt_words[i] != second_prompt_words[i]:
                return False
        return True

    def get_common_text(self, prompt: "PromptInfo") -> str:
        first_prompt_words = self.text.split(" ")
        second_prompt_words = prompt.text.split(" ")
        number_of_words = min(len(first_prompt_words), len(second_prompt_words))

        for i in range(number_of_words):
            if first_prompt_words[i] != second_prompt_words[i]:
                common_prompt = " ".join(first_prompt_words[:i])
                return common_prompt
        return " ".join(first_prompt_words[:number_of_words])


class TaskDetector:
    def __init__(self):
        # TODO: Add check for memory usage: i.e. if i am saving too many prompts
        self._prompt_list = []

        self._keywords_dict = {
            "TEXT_SUMMARIZATION": ["summarize", "summary", "summarization"],
            "TEXT_CLASSIFICATION": ["classify", "classification"],
            "CHAT": ["chat", "chatbot", "assistant"],
        }

    def detect_task_from_text(self, text: str) -> Task:
        prompt_info = PromptInfo(text=text, task=Task.UNDETECTED)
        current_prompt = self._get_prompt_from_prompt_list(prompt_info)
        self._assign_prompt_task(current_prompt)
        return current_prompt.task

    def _assign_prompt_task(self, prompt: PromptInfo):
        if prompt.task != Task.UNDETECTED:
            return
        for task_string, keywords in self._keywords_dict.items():
            for keyword in keywords:
                if keyword in prompt.text:
                    prompt.task = Task[task_string]
        if prompt.task == Task.UNDETECTED:
            prompt.task = Task.TEXT_GENERATION

    def _get_prompt_from_prompt_list(self, prompt: PromptInfo) -> PromptInfo:
        matching_prompt = self._get_matching_prompt_from_prompt_list(prompt)
        if matching_prompt is not None:
            common_prompt = matching_prompt.get_common_text(prompt)
            matching_prompt.text = common_prompt
            return matching_prompt
        else:
            self._prompt_list.append(prompt)
            return prompt

    def _get_matching_prompt_from_prompt_list(
        self, prompt: PromptInfo
    ) -> Optional[PromptInfo]:
        for p in self._prompt_list:
            if p.is_matching(prompt):
                return p
        return None
