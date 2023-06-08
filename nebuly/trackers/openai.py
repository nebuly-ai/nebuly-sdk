import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Dict, Tuple, Optional, Any, Callable, Generator

import openai
import tiktoken as tiktoken

from nebuly.core.core import DataPackageConverter, NebulyQueue, QueueObject, Tracker
from nebuly.core.schemas import (
    GenericProviderAttributes,
    NebulyDataPackage,
    Provider,
    TagData,
    Task,
    RawTrackedData,
)
from nebuly.utils.functions import (
    get_current_timestamp,
    get_media_file_length_in_seconds,
    transform_args_to_kwargs,
)

nebuly_logger = logging.getLogger(name=__name__)
nebuly_logger.setLevel(level=logging.INFO)


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


class OpenAIAttributes(GenericProviderAttributes):
    api_type: OpenAIAPIType
    api_key: str
    timestamp_openai: Optional[int] = None

    model: Optional[str] = None
    n_prompt_tokens: Optional[int] = None
    n_completion_tokens: Optional[int] = None

    n_output_images: Optional[int] = None
    image_size: Optional[str] = None

    audio_duration_seconds: Optional[int] = None

    training_file_id: Optional[str] = None
    training_id: Optional[str] = None


@dataclass
class OpenAIRawTrackedData(RawTrackedData):
    tag_data: TagData
    timestamp: float
    timestamp_end: float
    request_kwargs: Dict[str, Any]
    request_response: Dict[str, Any]
    api_type: OpenAIAPIType
    api_key: Optional[str]
    api_provider: str


class APITypeBodyFiller(ABC):
    @staticmethod
    @abstractmethod
    def fill_body_with_request_data(
        body: OpenAIAttributes,
        request_kwargs: Dict[str, Any],
        request_response: Dict[str, Any],
    ):
        raise NotImplementedError


class TextAPIBodyFiller(APITypeBodyFiller):
    @staticmethod
    def fill_body_with_request_data(
        body: OpenAIAttributes,
        request_kwargs: Dict[str, Any],
        request_response: Dict[str, Any],
    ):
        try:
            model = request_kwargs["model"]
            body.model = model
        except KeyError:
            model = None

        # Standard API Call
        try:
            body.n_prompt_tokens = int(request_response["usage"]["prompt_tokens"])
        except KeyError:
            pass
        try:
            body.n_completion_tokens = int(
                request_response["usage"]["completion_tokens"]
            )
        except KeyError:
            pass
        try:
            body.timestamp_openai = int(request_response["created"])
        except KeyError:
            pass

        # Stream API Call
        try:
            stream = request_kwargs["stream"]
        except KeyError:
            stream = False

        if stream is True:
            try:
                print(request_kwargs)
                if body.api_type == OpenAIAPIType.CHAT:
                    input_text = request_kwargs["messages"][0]["content"]
                elif body.api_type == OpenAIAPIType.TEXT_COMPLETION:
                    input_text = request_kwargs["prompt"]
                else:
                    input_text = ""
                body.n_prompt_tokens = TextAPIBodyFiller._num_tokens_from_string(
                    string=input_text, encoding_name=model
                )
            except KeyError:
                pass
            try:
                output_text = request_response["output_text"]
                body.n_completion_tokens = TextAPIBodyFiller._num_tokens_from_string(
                    string=output_text, encoding_name=model
                )
            except KeyError:
                pass
        return body

    @staticmethod
    def _num_tokens_from_string(string: str, encoding_name: str) -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.encoding_for_model(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens


class ImageAPIBodyFiller(APITypeBodyFiller):
    def fill_body_with_request_data(
        self,
        body: OpenAIAttributes,
        request_kwargs: Dict[str, Any],
        request_response: Dict[str, Any],
    ):
        try:
            body.model = request_kwargs["model"]
        except KeyError:
            body.model = "dall-e"
        try:
            body.training_file_id = request_kwargs["training_file"]
        except KeyError:
            pass
        try:
            body.training_id = request_response["id"]
        except KeyError:
            pass
        try:
            body.timestamp_openai = int(request_response["created_at"])
        except KeyError:
            pass


class AudioAPIBodyFiller(APITypeBodyFiller):
    def fill_body_with_request_data(
        self,
        body: OpenAIAttributes,
        request_kwargs: Dict[str, Any],
        request_response: Dict[str, Any],
    ):
        try:
            body.model = request_kwargs["model"]
        except KeyError:
            pass
        try:
            file_name: str = request_kwargs["file"].name
            body.audio_duration_seconds = get_media_file_length_in_seconds(
                file_path=file_name
            )
        except KeyError:
            pass


class FineTuneAPIBodyFiller(APITypeBodyFiller):
    def fill_body_with_request_data(
        self,
        body: OpenAIAttributes,
        request_kwargs: Dict[str, Any],
        request_response: Dict[str, Any],
    ):
        try:
            body.model = request_kwargs["model"]
        except KeyError:
            pass
        try:
            body.training_file_id = request_kwargs["training_file"]
        except KeyError:
            pass
        try:
            body.training_id = request_response["id"]
        except KeyError:
            pass
        try:
            body.timestamp_openai = int(request_response["created_at"])
        except KeyError:
            pass


class ModerationAPIBodyFiller(APITypeBodyFiller):
    def fill_body_with_request_data(
        self,
        body: OpenAIAttributes,
        request_kwargs: Dict[str, Any],
        request_response: Dict[str, Any],
    ):
        try:
            body.model = request_response["model"]
        except KeyError:
            pass
        # is free now 19/05/2023: I don't have usage data


OPENAI_FILLER_REGISTRY = {
    OpenAIAPIType.TEXT_COMPLETION: TextAPIBodyFiller(),
    OpenAIAPIType.CHAT: TextAPIBodyFiller(),
    OpenAIAPIType.EDIT: TextAPIBodyFiller(),
    OpenAIAPIType.EMBEDDING: TextAPIBodyFiller(),
    OpenAIAPIType.FINETUNE: FineTuneAPIBodyFiller(),
    OpenAIAPIType.MODERATION: ModerationAPIBodyFiller(),
    OpenAIAPIType.IMAGE_CREATE: ImageAPIBodyFiller(),
    OpenAIAPIType.IMAGE_EDIT: ImageAPIBodyFiller(),
    OpenAIAPIType.IMAGE_VARIATION: ImageAPIBodyFiller(),
    OpenAIAPIType.AUDIO_TRANSCRIBE: AudioAPIBodyFiller(),
    OpenAIAPIType.AUDIO_TRANSLATE: AudioAPIBodyFiller(),
}

OPENAI_TASK_REGISTRY = {
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

OPENAI_PROVIDER_REGISTRY = {
    "azure": Provider.AZURE_OPENAI,
    "open_ai": Provider.OPENAI,
}


class OpenAIDataPackageConverter(DataPackageConverter):
    def get_data_package(
        self,
        raw_data: OpenAIRawTrackedData,
    ) -> NebulyDataPackage:
        filler = OPENAI_FILLER_REGISTRY[raw_data.api_type]
        provider = OPENAI_PROVIDER_REGISTRY[raw_data.api_provider]
        detected_task = self._get_task(
            raw_data.tag_data,
            raw_data.api_type,
            raw_data.request_kwargs,
        )
        body = OpenAIAttributes(
            project=raw_data.tag_data.project,
            phase=raw_data.tag_data.phase,
            task=detected_task,
            api_type=raw_data.api_type,
            api_provider=provider,
            api_key=raw_data.api_key,
            timestamp=raw_data.timestamp,
            timestamp_end=raw_data.timestamp_end,
        )
        filler.fill_body_with_request_data(
            body=body,
            request_kwargs=raw_data.request_kwargs,
            request_response=raw_data.request_response,
        )
        return NebulyDataPackage(kind=provider, body=body)

    def _get_task(
        self,
        tag_data: TagData,
        api_type: OpenAIAPIType,
        request_kwargs: Dict[str, Any],
    ) -> Task:
        if tag_data.task != Task.UNKNOWN:
            return tag_data.task

        try:
            prompt = request_kwargs["prompt"][0]
            if api_type == OpenAIAPIType.TEXT_COMPLETION:
                return self._task_detector.detect_task_from_text(text=prompt)
            return OPENAI_TASK_REGISTRY[api_type]
        except KeyError:
            return OPENAI_TASK_REGISTRY[api_type]


class OpenAIQueueObject(QueueObject):
    def __init__(
        self,
        data_package_converter: OpenAIDataPackageConverter,
        request_kwargs: Dict[str, Any],
        request_response: Dict[str, Any],
        api_type: OpenAIAPIType,
        api_key: Optional[str],
        api_provider: str,
        timestamp: float,
        timestamp_end: float,
    ) -> None:
        super().__init__(data_package_converter)
        self._request_kwargs = request_kwargs
        self._request_response = request_response
        self._api_type = api_type
        self._api_key = api_key
        self._api_provider = api_provider
        self._timestamp = timestamp
        self._timestamp_end = timestamp_end

    def as_data_package(self) -> NebulyDataPackage:
        return self._data_package_converter.get_data_package(
            raw_data=OpenAIRawTrackedData(
                api_type=self._api_type,
                api_key=self._api_key,
                api_provider=self._api_provider,
                request_kwargs=self._request_kwargs,
                request_response=self._request_response,
                timestamp=self._timestamp,
                timestamp_end=self._timestamp_end,
                tag_data=self._tag_data,
            )
        )


class WrappingStrategy(ABC):
    @abstractmethod
    def wrap(
        self,
        nebuly_queue: NebulyQueue,
        original_method: Callable,
        request_kwargs: Dict[str, Any],
        api_type: OpenAIAPIType,
    ) -> Any:
        """Wrap the original method to capture the desired data"""
        pass


class APICallWrappingStrategy(WrappingStrategy):
    def wrap(
        self,
        nebuly_queue: NebulyQueue,
        original_method: Callable,
        request_kwargs: Dict[str, Any],
        api_type: OpenAIAPIType,
    ) -> Any:
        timestamp = get_current_timestamp()
        request_response = original_method(**request_kwargs)
        timestamp_end = get_current_timestamp()

        api_provider = openai.api_type
        api_key = openai.api_key

        queue_object = OpenAIQueueObject(
            data_package_converter=OpenAIDataPackageConverter(),
            request_kwargs=request_kwargs,
            request_response=request_response,
            api_type=api_type,
            api_key=api_key,
            api_provider=api_provider,
            timestamp=timestamp,
            timestamp_end=timestamp_end,
        )
        nebuly_queue.put(item=queue_object, timeout=0)

        return request_response


class GeneratorWrappingStrategy(WrappingStrategy, ABC):
    def __init__(self) -> None:
        self._nebuly_queue = None
        self._timestamp = None
        self._api_provider = None
        self._api_key = None
        self._api_type = None
        self._request_kwargs = None

    def wrap(
        self,
        nebuly_queue: NebulyQueue,
        original_method: Callable,
        request_kwargs: Dict[str, Any],
        api_type: OpenAIAPIType,
    ) -> Any:
        self._nebuly_queue = nebuly_queue
        self._timestamp = get_current_timestamp()
        request_response = original_method(**request_kwargs)
        self._api_provider = openai.api_type
        self._api_key = openai.api_key
        self._request_kwargs = request_kwargs
        self._api_type = api_type

        wrapped_response = self._create_wrapped_generator(request_response)
        return wrapped_response

    def _create_wrapped_generator(self, request_response: Generator) -> Generator:
        output_text_list = []

        def wrapped_generator():
            for element in request_response:
                self._track_generator_element(element, output_text_list)
                yield element

            timestamp_end = get_current_timestamp()
            output_text = "".join(output_text_list)
            mocked_request_response = {
                "output_text": output_text,
            }
            queue_object = OpenAIQueueObject(
                data_package_converter=OpenAIDataPackageConverter(),
                request_kwargs=self._request_kwargs,
                request_response=mocked_request_response,
                api_type=self._api_type,
                api_key=self._api_key,
                api_provider=self._api_provider,
                timestamp=self._timestamp,
                timestamp_end=timestamp_end,
            )
            self._nebuly_queue.put(item=queue_object, timeout=0)

        return wrapped_generator()

    @abstractmethod
    def _track_generator_element(self, element: Any, output_text) -> None:
        ...


class TextCompletionGeneratorWrappingStrategy(GeneratorWrappingStrategy):
    def _track_generator_element(self, element: Any, output_text) -> None:
        try:
            text = element["choices"][0]["text"]
            output_text.append(text)
        except (KeyError, IndexError):
            pass


class ChatCompletionGeneratorWrappingStrategy(GeneratorWrappingStrategy):
    def _track_generator_element(self, element: Any, output_text) -> None:
        try:
            delta = element["choices"][0]["delta"]
        except (KeyError, IndexError):
            return
        try:
            text = delta["content"]
            output_text.append(text)
        except KeyError:
            pass


class OpenAITracker(Tracker):
    _original_completion_create = None
    _original_chat_completion_create = None
    _original_edit_create = None
    _original_image_create = None
    _original_image_edit = None
    _original_image_variation = None
    _original_embedding_create = None
    _original_audio_transcribe = None
    _original_audio_translate = None
    _original_moderation_create = None
    _original_finetune = None

    FROM_METHOD_TO_API_TYPE = {}

    def __init__(self, nebuly_queue: NebulyQueue) -> None:
        self._nebuly_queue: NebulyQueue = nebuly_queue

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
        self.FROM_METHOD_TO_API_TYPE = {
            self._original_completion_create: OpenAIAPIType.TEXT_COMPLETION,
            self._original_chat_completion_create: OpenAIAPIType.CHAT,
            self._original_edit_create: OpenAIAPIType.EDIT,
            self._original_image_create: OpenAIAPIType.IMAGE_CREATE,
            self._original_image_edit: OpenAIAPIType.IMAGE_EDIT,
            self._original_image_variation: OpenAIAPIType.IMAGE_VARIATION,
            self._original_embedding_create: OpenAIAPIType.EMBEDDING,
            self._original_audio_transcribe: OpenAIAPIType.AUDIO_TRANSCRIBE,
            self._original_audio_translate: OpenAIAPIType.AUDIO_TRANSLATE,
            self._original_moderation_create: OpenAIAPIType.MODERATION,
            self._original_finetune: OpenAIAPIType.FINETUNE,
        }

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
        *request_args: Tuple,
        **request_kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        request_kwargs = transform_args_to_kwargs(
            func=original_method,
            func_args=request_args,
            func_kwargs=request_kwargs,
            specific_keyword="params",
        )
        api_type = self.FROM_METHOD_TO_API_TYPE[original_method]

        wrap_strategy = APICallWrappingStrategy()
        try:
            if request_kwargs["stream"] is True:
                if api_type == OpenAIAPIType.CHAT:
                    wrap_strategy = ChatCompletionGeneratorWrappingStrategy()
                elif api_type == OpenAIAPIType.TEXT_COMPLETION:
                    wrap_strategy = TextCompletionGeneratorWrappingStrategy()
        except KeyError:
            pass

        request_response = wrap_strategy.wrap(
            nebuly_queue=self._nebuly_queue,
            original_method=original_method,
            request_kwargs=request_kwargs,
            api_type=api_type,
        )
        return request_response
