from enum import Enum
from functools import partial
from typing import Dict, Tuple, Optional, Any

from nebuly.core.queues import NebulyQueue, DataPackageConverter, QueueObject, Tracker
from nebuly.core.schemas import (
    Provider,
    Task,
    NebulyDataPackage,
    TagData,
    GenericProviderAttributes,
)
from nebuly.utils.functions import (
    transform_args_to_kwargs,
    get_media_file_length_in_seconds,
    get_current_timestamp,
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


APITYPE_TO_TASK_DICT: dict[OpenAIAPIType, Task] = {
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


OPENAI_PROVIDER_DICT: dict[str, Provider] = {
    "azure": Provider.AZURE_OPENAI,
    "open_ai": Provider.OPENAI,
}


class OpenAIAttributes(GenericProviderAttributes):
    api_type: str
    timestamp_openai: Optional[int] = None

    model: Optional[str] = None
    n_prompt_tokens: Optional[int] = None
    n_output_tokens: Optional[int] = None

    n_output_images: Optional[int] = None
    image_size: Optional[str] = None

    audio_duration_seconds: Optional[int] = None

    training_file_id: Optional[str] = None
    training_id: Optional[str] = None


class OpenAIDataPackageConverter(DataPackageConverter):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self._provider: Provider = OPENAI_PROVIDER_DICT[openai.api_type]

    def get_data_package(
        self,
        tag_data: TagData,
        request_kwargs: Dict[str, Any],
        request_response: Dict[str, Any],
        api_type: str,
        timestamp: float,
        timestamp_end: float,
    ) -> NebulyDataPackage:
        """Converts the QueueObject into a NebulyDataPackage object using the
        data from the OpenAI API response, the request kwargs, the api type,
        and the TagData attributes assigned to the QueueObject by the
        NebulyQueue.

        Args:
            tag_data (TagData): The TagData object assigned to the QueueObject
            request_kwargs (Dict[str, Any]): The request kwargs used to make the
                OpenAI API request
            request_response (Dict[str, Any]): The response from the OpenAI API
            api_type (str): The type of OpenAI API request
            timestamp (float): The SDK timestamp captured just before the issuing the
                request to OpenAI.
            timestamp_end (float): The SDK timestamp captured just after the issuing the
                request to OpenAI.

        Returns:
            NebulyDataPackage: The NebulyDataPackage that can be sent to the Nebuly
                Server.
        """

        model: str | None = None
        n_input_tokens: int | None = None
        n_completion_tokens: int | None = None
        image_size: str | None = None
        n_output_images: int | None = None
        audio_duration_seconds: int | None = None
        training_file_id: str | None = None
        training_id: str | None = None
        timestamp_openai: int | None = None

        detected_task: Task = self._get_task(
            tag_data=tag_data,
            api_type=OpenAIAPIType(value=api_type),
            request_kwargs=request_kwargs,
        )

        if (
            (api_type == OpenAIAPIType.TEXT_COMPLETION.value)
            or (api_type == OpenAIAPIType.CHAT.value)
            or (api_type == OpenAIAPIType.EMBEDDING.value)
            or (api_type == OpenAIAPIType.EDIT.value)
        ):
            (
                model,
                n_input_tokens,
                n_completion_tokens,
                timestamp_openai,
            ) = self._get_text_api_data(
                request_kwargs=request_kwargs, request_response=request_response
            )
        elif (
            (api_type == OpenAIAPIType.IMAGE_CREATE.value)
            or (api_type == OpenAIAPIType.IMAGE_EDIT.value)
            or (api_type == OpenAIAPIType.IMAGE_VARIATION.value)
        ):
            (
                n_output_images,
                image_size,
                timestamp_openai,
            ) = self._get_image_api_data(
                request_kwargs=request_kwargs, request_response=request_response
            )
        elif (api_type == OpenAIAPIType.AUDIO_TRANSCRIBE.value) or (
            api_type == OpenAIAPIType.AUDIO_TRANSLATE.value
        ):
            (
                model,
                audio_duration_seconds,
            ) = self._get_voice_request_data(request_kwargs=request_kwargs)
        elif api_type == OpenAIAPIType.FINETUNE.value:
            (
                model,
                training_file_id,
                training_id,
                timestamp_openai,
            ) = self._get_finetune_request_data(
                request_kwargs=request_kwargs, request_response=request_response
            )
        elif api_type == OpenAIAPIType.MODERATION.value:
            model = self._get_moderation_request_data(request_response=request_response)
        else:
            nebuly_logger.error(msg=f"Unknown OpenAI API type: {api_type}")

        nebuly_body: OpenAIAttributes = OpenAIAttributes(
            project=tag_data.project,
            phase=tag_data.phase,
            task=detected_task,
            api_type=api_type,
            timestamp=timestamp,
            timestamp_end=timestamp_end,
            timestamp_openai=timestamp_openai,
            model=model,
            n_prompt_tokens=n_input_tokens,
            n_output_tokens=n_completion_tokens,
            n_output_images=n_output_images,
            image_size=image_size,
            audio_duration_seconds=audio_duration_seconds,
            training_file_id=training_file_id,
            training_id=training_id,
        )

        return NebulyDataPackage(
            kind=self._provider,
            body=nebuly_body,
        )

    def _get_task(
        self,
        tag_data: TagData,
        api_type: OpenAIAPIType,
        request_kwargs: Dict[str, Any],
    ) -> Task:
        if tag_data.task != Task.UNKNOWN:
            return tag_data.task
        if (api_type == OpenAIAPIType.TEXT_COMPLETION) and (
            "prompt" in request_kwargs.keys()
        ):
            return self._task_detector.detect_task_from_text(
                text=request_kwargs["prompt"][0]
            )
        else:
            return APITYPE_TO_TASK_DICT[api_type]

    def _get_text_api_data(
        self, request_kwargs: Dict[str, Any], request_response: Dict[str, Any]
    ) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[int]]:
        model: str | None = None
        if "model" in request_kwargs.keys():
            model = request_kwargs["model"]

        n_prompt_tokens: int | None = None
        if "usage" in request_response.keys():
            if "prompt_tokens" in request_response["usage"].keys():
                n_prompt_tokens = int(request_response["usage"]["prompt_tokens"])

        n_completion_tokens: int | None = None
        if "usage" in request_response.keys():
            if "prompt_tokens" in request_response["usage"].keys():
                n_completion_tokens = int(
                    request_response["usage"]["completion_tokens"]
                )  # noqa 501

        timestamp_openai: int | None = None
        if "created" in request_response.keys():
            timestamp_openai = int(request_response["created"])

        return model, n_prompt_tokens, n_completion_tokens, timestamp_openai

    def _get_image_api_data(
        self, request_kwargs: Dict[str, Any], request_response: Dict[str, Any]
    ) -> Tuple[Optional[int], Optional[str], Optional[int]]:
        number_of_images: int | None = None
        if "n" in request_kwargs.keys():
            number_of_images = int(request_kwargs["n"])

        image_size: str | None = None
        if "size" in request_kwargs.keys():
            image_size = request_kwargs["size"]

        timestamp_openai: int | None = None
        if "created" in request_response.keys():
            timestamp_openai = int(request_response["created"])
        return number_of_images, image_size, timestamp_openai

    def _get_voice_request_data(
        self,
        request_kwargs: Dict[str, Any],
    ) -> Tuple[Optional[str], Optional[int]]:
        model: str | None = None
        if "model" in request_kwargs.keys():
            model = request_kwargs["model"]

        audio_duration_seconds: int | None = None
        if "file" in request_kwargs.keys():
            file_name: str = request_kwargs["file"].name
            audio_duration_seconds = get_media_file_length_in_seconds(
                file_path=file_name
            )

        return model, audio_duration_seconds

    def _get_finetune_request_data(
        self,
        request_kwargs: Dict[str, Any],
        request_response: Dict[str, Any],
    ) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[int]]:
        model: str | None = None
        if "model" in request_kwargs.keys():
            model = request_kwargs["model"]

        training_file_id: str | None = None
        if "training_file" in request_kwargs.keys():
            training_file_id = request_kwargs["training_file"]

        training_id: str | None = None
        if "id" in request_response.keys():
            training_id = request_response["id"]

        timestamp_openai: int | None = None
        if "created_at" in request_response.keys():
            timestamp_openai = int(request_response["created_at"])

        return model, training_file_id, training_id, timestamp_openai

    def _get_moderation_request_data(
        self, request_response: Dict[str, Any]
    ) -> Optional[str]:
        model: str | None = None
        if "model" in request_response.keys():
            model = request_response["model"]
        # is free now 19/05/2023: I don't have usage data
        return model


class OpenAIQueueObject(QueueObject):
    def __init__(
        self,
        data_package_converter: DataPackageConverter,
        request_kwargs: Dict[str, Any],
        request_response: Dict[str, Any],
        api_type: str,
        timestamp: float,
        timestamp_end: float,
    ) -> None:
        super().__init__()

        self._api_type: str = api_type
        self._data_package_converter: DataPackageConverter = data_package_converter
        self._request_kwargs: Dict[str, Any] = request_kwargs
        self._request_response: Dict[str, Any] = request_response
        self._timestamp: float = timestamp
        self._timestamp_end: float = timestamp_end

    def as_data_package(self) -> NebulyDataPackage:
        return self._data_package_converter.get_data_package(
            tag_data=self._tag_data,
            request_kwargs=self._request_kwargs,
            request_response=self._request_response,
            api_type=self._api_type,
            timestamp=self._timestamp,
            timestamp_end=self._timestamp_end,
        )


class OpenAITracker(Tracker):
    def __init__(self, nebuly_queue: NebulyQueue) -> None:
        self._nebuly_queue: NebulyQueue = nebuly_queue

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

    def replace_sdk_functions(self) -> None:
        """Replace OpenAI SDK functions with custom ones."""
        self._replace_text_completion()
        self._replace_chat_completion()
        self._replace_edit()
        self._replace_image()
        self._replace_embedding()
        self._replace_audio()
        self._replace_finetune()
        self._replace_moderation()

    def _replace_text_completion(self) -> None:
        self._original_completion_create = openai.Completion.create
        new_tracked_method = partial(
            self._track_method, self._original_completion_create
        )
        openai.Completion.create = new_tracked_method

    def _replace_chat_completion(self) -> None:
        self._original_chat_completion_create = openai.ChatCompletion.create  # noqa 501
        new_tracked_method = partial(
            self._track_method, self._original_chat_completion_create
        )
        openai.ChatCompletion.create = new_tracked_method

    def _replace_edit(self) -> None:
        self._original_edit_create = openai.Edit.create
        new_tracked_method = partial(
            self._track_method,
            self._original_edit_create,
        )
        openai.Edit.create = new_tracked_method

    def _replace_image(self) -> None:
        self._original_image_create = openai.Image.create
        new_tracked_method = partial(
            self._track_method,
            self._original_image_create,
        )
        openai.Image.create = new_tracked_method

        self._original_image_edit = openai.Image.create_edit
        new_tracked_method = partial(
            self._track_method,
            self._original_image_edit,
        )
        openai.Image.create_edit = new_tracked_method

        self._original_image_variation = openai.Image.create_variation
        new_tracked_method = partial(
            self._track_method,
            self._original_image_variation,
        )
        openai.Image.create_variation = new_tracked_method

    def _replace_embedding(self) -> None:
        self._original_embedding_create = openai.Embedding.create
        new_tracked_method = partial(
            self._track_method,
            self._original_embedding_create,
        )
        openai.Embedding.create = new_tracked_method

    def _replace_audio(self) -> None:
        self._original_audio_transcribe = openai.Audio.transcribe
        new_tracked_method = partial(
            self._track_method,
            self._original_audio_transcribe,
        )
        openai.Audio.transcribe = new_tracked_method

        self._original_audio_translate = openai.Audio.translate
        new_tracked_method = partial(
            self._track_method,
            self._original_audio_translate,
        )
        openai.Audio.translate = new_tracked_method

    def _replace_finetune(self) -> None:
        self._original_finetune = openai.FineTune.create
        new_tracked_method = partial(
            self._track_method,
            self._original_finetune,
        )
        openai.FineTune.create = new_tracked_method

    def _replace_moderation(self) -> None:
        self._original_moderation_create = openai.Moderation.create
        new_tracked_method = partial(
            self._track_method,
            self._original_moderation_create,
        )
        openai.Moderation.create = new_tracked_method

    def _track_method(
        self,
        original_method: Any,
        *request_args: Dict[str, Any],
        **request_kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        timestamp: float = get_current_timestamp()
        request_response: Dict[str, Any] = original_method(
            *request_args, **request_kwargs
        )
        timestamp_end: float = get_current_timestamp()

        request_kwargs = transform_args_to_kwargs(
            func=original_method, func_args=request_args, func_kwargs=request_kwargs
        )
        api_type: OpenAIAPIType = self._assign_api_type(original_method=original_method)
        self._track_openai_api(
            request_kwargs=request_kwargs,
            request_response=request_response,
            api_type=api_type,
            timestamp=timestamp,
            timestamp_end=timestamp_end,
        )
        return request_response

    def _assign_api_type(self, original_method: Any) -> OpenAIAPIType:
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
        request_kwargs: Dict[str, Any],
        request_response: Dict[str, Any],
        api_type: OpenAIAPIType,
        timestamp: float,
        timestamp_end: float,
    ) -> None:
        queue_object = OpenAIQueueObject(
            data_package_converter=OpenAIDataPackageConverter(),
            request_kwargs=request_kwargs,
            request_response=request_response,
            api_type=api_type.value,
            timestamp=timestamp,
            timestamp_end=timestamp_end,
        )
        self._nebuly_queue.put(item=queue_object, timeout=0)
