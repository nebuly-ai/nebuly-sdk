from enum import Enum
from functools import partial
from typing import Dict, Tuple, Optional

from nebuly.core.queues import NebulyQueue, DataPackageConverter, QueueObject
from nebuly.core.schemas import Provider, Task, NebulyDataPackage
from nebuly.utils.functions import (
    transform_args_to_kwargs,
    get_media_file_length_in_seconds,
)
from nebuly.utils.logger import nebuly_logger

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
    MODERATION = "moderation"
    UNKNOWN = None


APITYPE_TO_TASK_DICT = {
    OpenAIAPIType.CHAT: Task.CHAT,
    OpenAIAPIType.EDIT: Task.TEXT_EDITING,
    OpenAIAPIType.IMAGE_CREATE: Task.IMAGE_GENERATION,
    OpenAIAPIType.IMAGE_EDIT: Task.IMAGE_EDITING,
    OpenAIAPIType.IMAGE_VARIATION: Task.IMAGE_VARIATION,
    OpenAIAPIType.AUDIO_TRANSCRIBE: Task.AUDIO_TRANSCRIPTION,
    OpenAIAPIType.AUDIO_TRANSLATE: Task.AUDIO_TRANSLATION,
    OpenAIAPIType.EMBEDDING: Task.TEXT_EMBEDDING,
    OpenAIAPIType.FINETUNE: Task.FINETUNING,
    OpenAIAPIType.MODERATION: Task.TEXT_MODERATION,
    OpenAIAPIType.TEXT_COMPLETION: Task.TEXT_GENERATION,
}


OPENAI_PROVIDER_DICT = {"azure": Provider.AZURE_OPENAI, "open_ai": Provider.OPENAI}


class OpenaiAIDataPackageConverter(DataPackageConverter):
    def __init__(
        self,
        request_kwargs: Dict,
        request_response: Dict,
        api_type: OpenAIAPIType,
    ):
        super().__init__()
        self._provider = OPENAI_PROVIDER_DICT[openai.api_type]
        self._request_kwargs = request_kwargs
        self._request_response = request_response
        self._api_type = api_type

    def get_data_package(
        self,
    ) -> NebulyDataPackage:
        """Converts the QueueObject into a NebulyDataPackage object using the
        data from the OpenAI API response, the request kwargs, the api type,
        and the TaggedData attributes assigned to the QueueObject by the
        NebulyQueue.

        Returns:
            NebulyDataPackage: The NebulyDataPackage representation of the QueueObject
        """

        model_name = None
        prompt_tokens = None
        completion_tokens = None
        number_of_images = None
        image_size = None
        duration_in_seconds = None
        training_file_id = None

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
            (
                model_name,
                training_file_id,
            ) = self._get_finetune_request_data()
        elif self._api_type == OpenAIAPIType.MODERATION:
            (model_name) = self._get_moderation_request_data()
        else:
            nebuly_logger.error(f"Unknown OpenAI API type: {self._api_type}")

        return NebulyDataPackage(
            project=self._tag_data.project,
            phase=self._tag_data.phase,
            task=self._tag_data.task,
            api_type=self._api_type.value,
            provider=self._provider,
            timestamp=self._timestamp,
            model=model_name,
            n_input_tokens=prompt_tokens,
            n_output_tokens=completion_tokens,
            n_output_images=number_of_images,
            image_size=image_size,
            audio_duration_seconds=duration_in_seconds,
            training_file_id=training_file_id,
        )

    def _assign_task(self):
        if self._tag_data.task != Task.UNDETECTED:
            return
        if (self._api_type == OpenAIAPIType.TEXT_COMPLETION) and (
            "prompt" in self._request_kwargs.keys()
        ):
            self._tag_data.task = self._task_detector.detect_task_from_text(
                self._request_kwargs["prompt"][0]
            )
        else:
            self._tag_data.task = APITYPE_TO_TASK_DICT[self._api_type]

    def _get_text_api_data(self) -> Tuple[Optional[str], Optional[int], Optional[int]]:
        model_name = None
        if "model" in self._request_kwargs.keys():
            model_name = self._request_kwargs["model"]

        prompt_tokens = None
        if "usage" in self._request_response.keys():
            if "prompt_tokens" in self._request_response["usage"].keys():
                prompt_tokens = int(self._request_response["usage"]["prompt_tokens"])

        completion_tokens = None
        if "usage" in self._request_response.keys():
            if "prompt_tokens" in self._request_response["usage"].keys():
                completion_tokens = int(
                    self._request_response["usage"]["completion_tokens"]
                )  # noqa 501
        return model_name, prompt_tokens, completion_tokens

    def _get_image_api_data(self) -> Tuple[Optional[int], Optional[str]]:
        number_of_images = None
        if "n" in self._request_kwargs.keys():
            number_of_images = int(self._request_kwargs["n"])

        image_size = None
        if "size" in self._request_kwargs.keys():
            image_size = self._request_kwargs["size"]
        return number_of_images, image_size

    def _get_voice_request_data(self) -> Tuple[Optional[str], Optional[int]]:
        model_name = None
        if "model" in self._request_kwargs.keys():
            model_name = self._request_kwargs["model"]

        duration_in_seconds = None
        if "file" in self._request_kwargs.keys():
            file_name = self._request_kwargs["file"].name
            duration_in_seconds = get_media_file_length_in_seconds(file_name)

        return model_name, duration_in_seconds

    def _get_finetune_request_data(self) -> Tuple[Optional[str], Optional[str]]:
        model_name = None
        if "model" in self._request_kwargs.keys():
            model_name = self._request_kwargs["model"]

        training_file_id = None
        if "training_file" in self._request_kwargs.keys():
            training_file_id = self._request_kwargs["training_file"]

        return model_name, training_file_id

    def _get_moderation_request_data(self) -> Optional[str]:
        model_name = None
        if "model" in self._request_response.keys():
            model_name = self._request_response["model"]
        # TODO: price for this API is not clear yet, since for now this API is in beta,
        # and delivered for free.
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
        self._original_moderation_create = None

    def replace_sdk_functions(self):
        """Replace OpenAI SDK functions with custom ones."""
        self._replace_text_completion()
        self._replace_chat_completion()
        self._replace_edit()
        self._replace_image()
        self._replace_embedding()
        self._replace_audio()
        self._replace_finetune()
        self._replace_moderation()

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

    def _replace_moderation(self):
        self._original_moderation_create = openai.Moderation.create
        new_tracked_method = partial(
            self._track_method, self._original_moderation_create
        )
        openai.Moderation.create = new_tracked_method

    def _track_method(
        self, original_method: callable, *request_args, **request_kwargs
    ) -> Dict:
        # TODO: still missing the case of the request failing.
        request_response = original_method(*request_args, **request_kwargs)
        request_kwargs = transform_args_to_kwargs(
            original_method, request_args, request_kwargs
        )
        api_type = self._assign_api_type(original_method)
        self._track_openai_api(request_kwargs, request_response, api_type)
        return request_response

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
        elif original_method == self._original_moderation_create:
            return OpenAIAPIType.MODERATION
        else:
            return OpenAIAPIType.UNKNOWN

    def _track_openai_api(
        self,
        request_kwargs: Dict,
        request_response: Dict,
        api_type: OpenAIAPIType,
    ):
        queue_object = QueueObject(
            data_package_converter=OpenaiAIDataPackageConverter(
                request_kwargs=request_kwargs,
                request_response=request_response,
                api_type=api_type,
            ),
        )
        self._nebuly_queue.put(queue_object)
