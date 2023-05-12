from enum import Enum
from functools import partial
from typing import Dict, Tuple

from nebuly.core.nebuly_client import NebulyQueue, QueueObject
from nebuly.core.schemas import Provider, Task, NebulyDataPackage
from nebuly.utils.functions import transform_args_to_kwargs
from nebuly.utils.nebuly_logger import nebuly_logger

import openai


class OpenAIAPIType(Enum):
    TEXT_COMPLETION = "text_completion"
    CHAT = "chat"
    EDIT = "edit"
    IMAGE_CREATE = "image_create"
    IMAGE_EDIT = "image_edit"
    IMAGE_VARIATION = "image_variation"
    AUDIO_TRANSCRIBE = "audio_transcribe"
    AUDIO_TRANSLATE = "audio_translate"
    EMBEDDING = "embedding"
    FINETUNE = "finetune"
    UNKNOWN = None


class OpenAIQueueObject(QueueObject):
    def __init__(self, parameters: Dict, response: Dict, api_type: OpenAIAPIType):
        super().__init__(parameters=parameters, response=response)
        self._provider = Provider.OPENAI
        self._api_type = api_type

    def get_request_data(
        self,
    ) -> NebulyDataPackage:
        super().get_request_data()
        model_name = None
        prompt_tokens = None
        completion_tokens = None
        number_of_images = None
        image_size = None
        duration_in_seconds = None

        self._assign_task()

        if (
            (self._api_type == OpenAIAPIType.TEXT_COMPLETION)
            or (self._api_type == OpenAIAPIType.CHAT)
            or (self._api_type == OpenAIAPIType.EMBEDDING)
            or (self._api_type == OpenAIAPIType.EDIT)
        ):
            (
                model_name,
                prompt_tokens,
                completion_tokens,
            ) = self._get_text_api_data()
        elif (
            (self._api_type == OpenAIAPIType.IMAGE_CREATE)
            or (self._api_type == OpenAIAPIType.IMAGE_EDIT)
            or (self._api_type == OpenAIAPIType.IMAGE_VARIATION)
        ):
            (
                number_of_images,
                image_size,
            ) = self._get_image_api_data()
        elif (self._api_type == OpenAIAPIType.AUDIO_TRANSCRIBE) or (
            self._api_type == OpenAIAPIType.AUDIO_TRANSLATE
        ):
            (
                model_name,
                duration_in_seconds,
            ) = self._get_voice_request_data()
        elif self._api_type == OpenAIAPIType.FINETUNE:
            (model_name) = self._get_finetune_request_data()
        else:
            nebuly_logger.error("Unknown OpenAI API type")

        return NebulyDataPackage(
            project=self._project,
            phase=self._phase.value,
            task=self._task.value,
            api_type=self._api_type.value,
            provider=self._provider.value,
            timestamp=self._timestamp,
            model_name=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            number_of_images=number_of_images,
            image_size=image_size,
            duration_in_seconds=duration_in_seconds,
        )

    def _assign_task(
        self,
    ):
        if self._task != Task.UNDETECTED:
            return
        elif self._api_type == OpenAIAPIType.CHAT:
            self._task = Task.CHAT
        elif self._api_type == OpenAIAPIType.EDIT:
            self._task = Task.TEXT_EDITING
        elif self._api_type == OpenAIAPIType.IMAGE_CREATE:
            self._task = Task.IMAGE_GENERATION
        elif self._api_type == OpenAIAPIType.IMAGE_EDIT:
            self._task = Task.IMAGE_EDITING
        elif self._api_type == OpenAIAPIType.IMAGE_VARIATION:
            self._task = Task.IMAGE_VARIATION
        elif self._api_type == OpenAIAPIType.AUDIO_TRANSCRIBE:
            self._task = Task.AUDIO_TRANSCRIPTION
        elif self._api_type == OpenAIAPIType.AUDIO_TRANSLATE:
            self._task = Task.AUDIO_TRANSLATION
        elif self._api_type == OpenAIAPIType.EMBEDDING:
            self._task = Task.TEXT_EMBEDDING
        elif self._api_type == OpenAIAPIType.FINETUNE:
            self._task = Task.FINETUNING
        elif self._api_type == OpenAIAPIType.TEXT_COMPLETION:
            if "prompt" in self._parameters.keys():
                self._task = self.detect_task_from_text(self._parameters["prompt"][0])
            else:
                self._task = Task.TEXT_GENERATION

    def _get_text_api_data(self) -> Tuple[str, int, int]:
        model_name = "undetected"
        if "model" in self._parameters.keys():
            model_name = self._parameters["model"]

        prompt_tokens = -1
        if "usage" in self._response.keys():
            if "prompt_tokens" in self._response["usage"].keys():
                prompt_tokens = int(self._response["usage"]["prompt_tokens"])

        completion_tokens = -1
        if "usage" in self._response.keys():
            if "prompt_tokens" in self._response["usage"].keys():
                completion_tokens = int(self._response["usage"]["completion_tokens"])
        return model_name, prompt_tokens, completion_tokens

    def _get_image_api_data(self) -> Tuple[int, str]:
        number_of_images = -1
        if "n" in self._parameters.keys():
            number_of_images = int(self._parameters["n"])

        image_size = "undetected"
        if "size" in self._parameters.keys():
            image_size = self._parameters["size"]
        return number_of_images, image_size

    def _get_voice_request_data(self) -> Tuple[str, int]:
        model_name = "undetected"
        if "model" in self._parameters.keys():
            model_name = self._parameters["model"]

        # TODO: duration_in_seconds is not trivial.
        # it may requires additional external libraries:
        # (https://www.geeksforgeeks.org/how-to-get-the-duration-of-audio-in-python/)
        # reference to openai docs:
        # https://platform.openai.com/docs/api-reference/audio
        duration_in_seconds = 0
        return model_name, duration_in_seconds

    def _get_finetune_request_data(self) -> str:
        model_name = "undetected"
        if "model" in self._parameters.keys():
            model_name = self._parameters["model"]

        # TODO: trainings tokens may not be available in the response.
        # openaidocs:
        # https://platform.openai.com/docs/api-reference/fine-tunes/create
        return model_name


class OpenAITracker:
    def __init__(self, nebuly_queue: NebulyQueue):
        self._nebuly_queue = nebuly_queue

        self._original_completion_create = None
        self._original_chat_completion_create = None
        self._original_edit_create = None
        self._original_image_create = None
        self._original_image_edit = None
        self._original_image_variation = None
        self._original_embedding_create = None
        self._original_audio_transcribe = None
        self._original_audio_translate = None

    def replace_sdk_functions(self):
        self._replace_text_completion()
        self._replace_chat_completion()
        self._replace_edit()
        self._replace_image()
        self._replace_embedding()
        self._replace_audio()
        self._replace_finetune()

    def _replace_text_completion(self):
        self._original_completion_create = openai.Completion.create
        new_tracked_method = partial(
            self._track_method, self._original_completion_create
        )
        openai.Completion.create = new_tracked_method

    def _replace_chat_completion(self):
        self._original_chat_completion_create = openai.ChatCompletion.create
        new_tracked_method = partial(
            self._track_method, self._original_chat_completion_create
        )
        openai.ChatCompletion.create = new_tracked_method

    def _replace_edit(self):
        self._original_edit_create = openai.Edit.create
        new_tracked_method = partial(self._track_method, self._original_edit_create)
        openai.Edit.create = new_tracked_method

    def _replace_image(self):
        self._original_image_create = openai.Image.create
        new_tracked_method = partial(self._track_method, self._original_image_create)
        openai.Image.create = new_tracked_method

        self._original_image_edit = openai.Image.create_edit
        new_tracked_method = partial(self._track_method, self._original_image_edit)
        openai.Image.create_edit = new_tracked_method

        self._original_image_variation = openai.Image.create_variation
        new_tracked_method = partial(self._track_method, self._original_image_variation)
        openai.Image.create_variation = new_tracked_method

    def _replace_embedding(self):
        self._original_embedding_create = openai.Embedding.create
        new_tracked_method = partial(
            self._track_method, self._original_embedding_create
        )
        openai.Embedding.create = new_tracked_method

    def _replace_audio(self):
        self._original_audio_transcribe = openai.Audio.transcribe
        new_tracked_method = partial(
            self._track_method, self._original_audio_transcribe
        )
        openai.Audio.transcribe = new_tracked_method

        self._original_audio_translate = openai.Audio.translate
        new_tracked_method = partial(self._track_method, self._original_audio_translate)
        openai.Audio.translate = new_tracked_method

    def _replace_finetune(self):
        self._original_finetune = openai.FineTune.create
        new_tracked_method = partial(self._track_method, self._original_finetune)
        openai.FineTune.create = new_tracked_method

    def _track_method(
        self, original_method: callable, *request_args, **request_kwargs
    ) -> Dict:
        # TODO: still missing the case
        response = original_method(*request_args, **request_kwargs)
        parameters = transform_args_to_kwargs(
            original_method, request_args, request_kwargs
        )
        api_type = self._assign_api_type(original_method)
        self._track_openai_api(parameters, response, api_type)
        return response

    def _assign_api_type(self, original_method: callable):
        if original_method == self._original_completion_create:
            return OpenAIAPIType.TEXT_COMPLETION
        elif original_method == self._original_chat_completion_create:
            return OpenAIAPIType.CHAT
        elif original_method == self._original_audio_transcribe:
            return OpenAIAPIType.AUDIO_TRANSCRIBE
        elif original_method == self._original_audio_translate:
            return OpenAIAPIType.AUDIO_TRANSLATE
        elif original_method == self._original_edit_create:
            return OpenAIAPIType.EDIT
        elif original_method == self._original_image_create:
            return OpenAIAPIType.IMAGE_CREATE
        elif original_method == self._original_image_edit:
            return OpenAIAPIType.IMAGE_EDIT
        elif original_method == self._original_image_variation:
            return OpenAIAPIType.IMAGE_VARIATION
        elif original_method == self._original_embedding_create:
            return OpenAIAPIType.EMBEDDING
        elif original_method == self._original_finetune:
            return OpenAIAPIType.FINETUNE
        else:
            return OpenAIAPIType.UNKNOWN

    def _track_openai_api(
        self, parameters: Dict, response: Dict, api_type: OpenAIAPIType
    ):
        queue_object = OpenAIQueueObject(
            parameters=parameters, response=response, api_type=api_type
        )
        self._nebuly_queue.put(queue_object)
