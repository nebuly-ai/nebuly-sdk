from enum import Enum
from functools import partial
from typing import Dict, Tuple, Optional

from nebuly.core.queues import NebulyQueue, DataPackageConverter, QueueObject
from nebuly.core.schemas import (
    Provider,
    Task,
    NebulyDataPackage,
    TagData,
    NebulyRequestParams,
)
from nebuly.utils.functions import (
    transform_args_to_kwargs,
    get_media_file_length_in_seconds,
)
from nebuly.utils.logger import nebuly_logger
from nebuly.utils.functions import get_current_timestamp

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


class OpenAIDataPackageConverter(DataPackageConverter):
    def __init__(
        self,
    ):
        super().__init__()
        self._provider = OPENAI_PROVIDER_DICT[openai.api_type]

    def get_data_package(
        self,
        tag_data: TagData,
        request_kwargs: Dict,
        request_response: Dict,
        api_type: OpenAIAPIType,
        timestamp: float,
    ) -> NebulyDataPackage:
        """Converts the QueueObject into a NebulyDataPackage object using the
        data from the OpenAI API response, the request kwargs, the api type,
        and the TaggedData attributes assigned to the QueueObject by the
        NebulyQueue.

        Args:
            tag_data (TagData): The TagData object assigned to the QueueObject
            request_kwargs (Dict): The request kwargs used to make the API call
            request_response (Dict): The response from the API call
            api_type (OpenAIAPIType): The type of API call made
            timestamp (float): The timestamp of the API call

        Returns:
            NebulyDataPackage: The NebulyDataPackage representation of the QueueObject
        """

        model = None
        n_input_tokens = None
        n_completion_tokens = None
        number_of_images = None
        n_output_images = None
        audio_duration_seconds = None
        training_file_id = None
        training_id = None

        detected_task = self._get_task(tag_data, api_type, request_kwargs)

        if (
            (api_type == OpenAIAPIType.TEXT_COMPLETION)
            or (api_type == OpenAIAPIType.CHAT)
            or (api_type == OpenAIAPIType.EMBEDDING)
            or (api_type == OpenAIAPIType.EDIT)
        ):
            (
                model,
                n_input_tokens,
                n_completion_tokens,
            ) = self._get_text_api_data(request_kwargs, request_response)
        elif (
            (api_type == OpenAIAPIType.IMAGE_CREATE)
            or (api_type == OpenAIAPIType.IMAGE_EDIT)
            or (api_type == OpenAIAPIType.IMAGE_VARIATION)
        ):
            (
                number_of_images,
                n_output_images,
            ) = self._get_image_api_data(request_kwargs)
        elif (api_type == OpenAIAPIType.AUDIO_TRANSCRIBE) or (
            api_type == OpenAIAPIType.AUDIO_TRANSLATE
        ):
            (
                model,
                audio_duration_seconds,
            ) = self._get_voice_request_data(request_kwargs)
        elif api_type == OpenAIAPIType.FINETUNE:
            (model, training_file_id, training_id) = self._get_finetune_request_data(
                request_kwargs, request_response
            )
        elif api_type == OpenAIAPIType.MODERATION:
            model = self._get_moderation_request_data(request_response)
        else:
            nebuly_logger.error(f"Unknown OpenAI API type: {api_type}")

        return NebulyDataPackage(
            project=tag_data.project,
            phase=tag_data.phase,
            task=detected_task,
            api_type=api_type.value,
            timestamp=timestamp,
            model=model,
            n_prompt_tokens=n_input_tokens,
            n_output_tokens=n_completion_tokens,
            n_output_images=number_of_images,
            image_size=n_output_images,
            audio_duration_seconds=audio_duration_seconds,
            training_file_id=training_file_id,
            training_id=training_id,
        )

    def get_request_params(self):
        return NebulyRequestParams(
            kind=self._provider,
        )

    def _get_task(
        self,
        tag_data: TagData,
        api_type: OpenAIAPIType,
        request_kwargs: Dict,
    ) -> Task:
        if tag_data.task != Task.UNDETECTED:
            return tag_data.task
        if (api_type == OpenAIAPIType.TEXT_COMPLETION) and (
            "prompt" in request_kwargs.keys()
        ):
            return self._task_detector.detect_task_from_text(
                request_kwargs["prompt"][0]
            )
        else:
            return APITYPE_TO_TASK_DICT[api_type]

    def _get_text_api_data(
        self, request_kwargs: Dict, request_response: Dict
    ) -> Tuple[Optional[str], Optional[int], Optional[int]]:
        model = None
        if "model" in request_kwargs.keys():
            model = request_kwargs["model"]

        n_prompt_tokens = None
        if "usage" in request_response.keys():
            if "prompt_tokens" in request_response["usage"].keys():
                n_prompt_tokens = int(request_response["usage"]["prompt_tokens"])

        n_completion_tokens = None
        if "usage" in request_response.keys():
            if "prompt_tokens" in request_response["usage"].keys():
                n_completion_tokens = int(
                    request_response["usage"]["completion_tokens"]
                )  # noqa 501
        return model, n_prompt_tokens, n_completion_tokens

    def _get_image_api_data(
        self,
        request_kwargs: Dict,
    ) -> Tuple[Optional[int], Optional[str]]:
        number_of_images = None
        if "n" in request_kwargs.keys():
            number_of_images = int(request_kwargs["n"])

        image_size = None
        if "size" in request_kwargs.keys():
            image_size = request_kwargs["size"]
        return number_of_images, image_size

    def _get_voice_request_data(
        self,
        request_kwargs: Dict,
    ) -> Tuple[Optional[str], Optional[int]]:
        model = None
        if "model" in request_kwargs.keys():
            model = request_kwargs["model"]

        audio_duration_seconds = None
        if "file" in request_kwargs.keys():
            file_name = request_kwargs["file"].name
            audio_duration_seconds = get_media_file_length_in_seconds(file_name)

        return model, audio_duration_seconds

    def _get_finetune_request_data(
        self,
        request_kwargs: Dict,
        request_response: Dict,
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        model = None
        if "model" in request_kwargs.keys():
            model = request_kwargs["model"]

        training_file_id = None
        if "training_file" in request_kwargs.keys():
            training_file_id = request_kwargs["training_file"]

        training_id = None
        if "training_id" in request_response.keys():
            training_id = request_response["training_id"]

        return model, training_file_id, training_id

    def _get_moderation_request_data(self, request_response: Dict) -> Optional[str]:
        model = None
        if "model" in request_response.keys():
            model = request_response["model"]
        # is free now 19/05/2023: I don't have usage data
        return model


class OpenAIQueueObject(QueueObject):
    def __init__(
        self,
        data_package_converter: DataPackageConverter,
        request_kwargs: Dict,
        request_response: Dict,
        api_type: str,
        timestamp: float,
    ):
        super().__init__()

        self._data_package_converter = data_package_converter
        self._request_kwargs = request_kwargs
        self._request_response = request_response
        self._api_type = api_type
        self._timestamp = timestamp

    def as_data_package(self) -> NebulyDataPackage:
        return self._data_package_converter.get_data_package(
            tag_data=self._tag_data,
            request_kwargs=self._request_kwargs,
            request_response=self._request_response,
            api_type=self._api_type,
            timestamp=self._timestamp,
        )

    def as_request_params(self) -> NebulyRequestParams:
        return self._data_package_converter.get_request_params()


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
        queue_object = OpenAIQueueObject(
            data_package_converter=OpenAIDataPackageConverter(),
            request_kwargs=request_kwargs,
            request_response=request_response,
            api_type=api_type,
            timestamp=get_current_timestamp(),
        )
        self._nebuly_queue.put(queue_object, timeout=0)
