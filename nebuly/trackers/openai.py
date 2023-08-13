import datetime as dt
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from functools import partial
from typing import Any, Callable, Dict, Generator, Optional, Tuple

import openai
import tiktoken as tiktoken

from nebuly.core.queues import (
    DataPackageConverter,
    NebulyQueue,
    QueueObject,
    RawTrackedData,
    Tracker,
)
from nebuly.core.schemas import (
    GenericProviderAttributes,
    NebulyDataPackage,
    Provider,
    TagData,
    Task,
)
from nebuly.utils.functions import (
    get_current_timestamp,
    get_media_file_length_in_seconds,
    transform_args_to_kwargs,
)

nebuly_logger = logging.getLogger(name=__name__)

ADDITIONAL_PROMPT_TOKENS_FOR_CHAT_COMPLETION_GENERATION = 0
ADDITIONAL_COMPLETION_TOKENS_FOR_CHAT_COMPLETION_GENERATION = 1


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
    organization: Optional[str] = None
    timestamp_openai: Optional[datetime] = None
    user: Optional[str] = None

    model: Optional[str] = None
    n_input_tokens: Optional[int] = None
    n_output_tokens: Optional[int] = None

    n_output_images: Optional[int] = None
    image_size: Optional[str] = None

    audio_duration_seconds: Optional[int] = None

    training_file_id: Optional[str] = None
    training_id: Optional[str] = None
    n_epochs: Optional[int] = None


@dataclass
class OpenAIRawTrackedData(RawTrackedData):
    timestamp: float
    timestamp_end: float
    request_kwargs: Dict[str, Any]
    request_response: Dict[str, Any]
    api_type: OpenAIAPIType
    api_key: Optional[str]
    api_provider: str
    organization: Optional[str]
    timestamp_openai: Optional[int] = None


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
        stream = request_kwargs.get("stream", False)

        if stream is False:
            TextAPIBodyFiller._fill_body_for_standard_api_call(
                body=body,
                request_kwargs=request_kwargs,
                request_response=request_response,
            )
        else:
            TextAPIBodyFiller._fill_body_for_stream_api_call(
                body=body,
                request_kwargs=request_kwargs,
                request_response=request_response,
            )

    @staticmethod
    def _fill_body_for_standard_api_call(
        body: OpenAIAttributes,
        request_kwargs: Dict[str, Any],
        request_response: Dict[str, Any],
    ):
        body.user = request_kwargs.get("user")
        if "model" in request_response:
            body.model = request_response["model"]
        else:
            body.model = request_kwargs.get("model")

        timestamp_openai = request_response.get("created")
        body.timestamp_openai = (
            datetime.fromtimestamp(timestamp_openai, tz=dt.timezone.utc)
            if timestamp_openai
            else None
        )

        n_prompt_tokens = request_response.get("usage", {}).get("prompt_tokens")
        body.n_input_tokens = int(n_prompt_tokens) if n_prompt_tokens else None

        n_completion_tokens = request_response.get("usage", {}).get("completion_tokens")
        body.n_output_tokens = int(n_completion_tokens) if n_completion_tokens else None

    @staticmethod
    def _fill_body_for_stream_api_call(
        body: OpenAIAttributes,
        request_kwargs: Dict[str, Any],
        request_response: Dict[str, Any],
    ):
        body.user = request_kwargs.get("user")
        if "model" in request_response:
            body.model = request_response["model"]
        else:
            body.model = request_kwargs.get("model")

        if body.model is None:
            return

        try:
            if body.api_type == OpenAIAPIType.CHAT:
                input_tokens = TextAPIBodyFiller._num_tokens_from_messages(
                    request_kwargs["messages"], model=body.model
                )
            elif body.api_type == OpenAIAPIType.TEXT_COMPLETION:
                input_tokens = TextAPIBodyFiller._num_tokens_from_text(
                    string=request_kwargs.get("prompt", ""), encoding_name=body.model
                )
            else:
                input_tokens = None
            body.n_input_tokens = input_tokens
        except (KeyError, IndexError):
            pass

        output_text = request_response.get("output_text")
        if output_text is not None:
            body.n_output_tokens = TextAPIBodyFiller._num_tokens_from_text(
                string=output_text, encoding_name=body.model
            )

        timestamp_openai = request_response.get("timestamp_openai")
        if timestamp_openai is not None:
            body.timestamp_openai = datetime.fromtimestamp(
                timestamp_openai, tz=dt.timezone.utc
            )

        if body.api_type == OpenAIAPIType.CHAT:
            # OpenAI adds some tokens to the ones provided to and by the user.
            if body.n_input_tokens is not None:
                body.n_input_tokens += (
                    ADDITIONAL_PROMPT_TOKENS_FOR_CHAT_COMPLETION_GENERATION
                )
            if body.n_output_tokens is not None:
                body.n_output_tokens += (
                    ADDITIONAL_COMPLETION_TOKENS_FOR_CHAT_COMPLETION_GENERATION
                )

    @staticmethod
    def _num_tokens_from_text(string: str, encoding_name: str) -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.encoding_for_model(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    @staticmethod
    def _num_tokens_from_messages(messages: list, model: str):
        """Returns the number of tokens used by a list of messages."""
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        num_tokens = 0
        for message in messages:
            num_tokens += (
                4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            )
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens


class ImageAPIBodyFiller(APITypeBodyFiller):
    def fill_body_with_request_data(
        self,
        body: OpenAIAttributes,
        request_kwargs: Dict[str, Any],
        request_response: Dict[str, Any],
    ):
        body.user = request_kwargs.get("user")
        body.model = request_kwargs.get("model", "dall-e")
        body.image_size = request_kwargs.get("size")

        n_output_images = request_kwargs.get("n")
        body.n_output_images = int(n_output_images) if n_output_images else None

        timestamp_openai = request_response.get("created")
        body.timestamp_openai = (
            datetime.fromtimestamp(timestamp_openai, tz=dt.timezone.utc)
            if timestamp_openai
            else None
        )


class AudioAPIBodyFiller(APITypeBodyFiller):
    def fill_body_with_request_data(
        self,
        body: OpenAIAttributes,
        request_kwargs: Dict[str, Any],
        request_response: Dict[str, Any],
    ):
        body.user = request_kwargs.get("user")
        body.model = request_kwargs.get("model")
        try:
            file_name = request_kwargs.get("file")
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
        body.user = request_kwargs.get("user")
        body.model = request_response.get("model")
        body.n_epochs = request_response.get("hyperparams", {}).get("n_epochs")
        body.training_file_id = request_kwargs.get("training_file")
        body.training_id = request_response.get("id")

        timestamp_openai = request_response.get("created_at")
        body.timestamp_openai = (
            datetime.fromtimestamp(timestamp_openai, tz=dt.timezone.utc)
            if timestamp_openai
            else None
        )


class ModerationAPIBodyFiller(APITypeBodyFiller):
    def fill_body_with_request_data(
        self,
        body: OpenAIAttributes,
        request_kwargs: Dict[str, Any],
        request_response: Dict[str, Any],
    ):
        body.user = request_kwargs.get("user")
        body.model = request_response.get("model")
        # is free now 19/05/2023: I don't have usage data


FROM_API_TO_FILLER = {
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

FROM_API_TO_TASK = {
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

PROVIDER_REGISTRY = {
    "azure": Provider.AZURE_OPENAI,
    "open_ai": Provider.OPENAI,
}


class OpenAIDataPackageConverter(DataPackageConverter):
    def get_data_package(
        self,
        raw_data: OpenAIRawTrackedData,
        tag_data: TagData,
    ) -> NebulyDataPackage:
        filler = FROM_API_TO_FILLER[raw_data.api_type]
        provider = PROVIDER_REGISTRY[raw_data.api_provider]
        detected_task = self._get_task(
            tag_data,
            raw_data.api_type,
            raw_data.request_kwargs,
        )
        body = OpenAIAttributes(
            project=tag_data.project,
            development_phase=tag_data.development_phase,
            task=detected_task,
            api_type=raw_data.api_type,
            api_key=raw_data.api_key,
            timestamp=datetime.fromtimestamp(raw_data.timestamp, tz=dt.timezone.utc),
            timestamp_end=datetime.fromtimestamp(
                raw_data.timestamp_end, tz=dt.timezone.utc
            ),
            organization=raw_data.organization,
        )
        filler.fill_body_with_request_data(
            body=body,
            request_kwargs=raw_data.request_kwargs,
            request_response=raw_data.request_response,
        )
        return NebulyDataPackage(provider=provider, body=body)

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
            return FROM_API_TO_TASK[api_type]
        except KeyError:
            return FROM_API_TO_TASK[api_type]


class OpenAIQueueObject(QueueObject):
    def __init__(
        self,
        raw_data: OpenAIRawTrackedData,
        data_package_converter: DataPackageConverter = OpenAIDataPackageConverter(),
    ) -> None:
        super().__init__(
            raw_data=raw_data, data_package_converter=data_package_converter
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

        raw_data = OpenAIRawTrackedData(
            request_kwargs=request_kwargs,
            request_response=request_response,
            api_type=api_type,
            api_key=openai.api_key,
            api_provider=openai.api_type,
            timestamp=timestamp,
            timestamp_end=timestamp_end,
            organization=openai.organization,
        )

        queue_object = OpenAIQueueObject(raw_data)
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
        self._request_kwargs = request_kwargs
        self._api_type = api_type

        wrapped_response = self._create_wrapped_generator(request_response)
        return wrapped_response

    def _create_wrapped_generator(self, request_response: Generator) -> Generator:
        # output_text_list = []

        def wrapped_generator():
            output_text = ""
            timestamp_openai = None
            used_model = None
            for element in request_response:
                (
                    output_text,
                    timestamp_openai,
                    used_model,
                ) = self._track_generator_element(
                    element, output_text, timestamp_openai, used_model
                )
                yield element

            timestamp_end = get_current_timestamp()
            mocked_request_response = {
                "output_text": output_text,
                "timestamp_openai": timestamp_openai,
                "model": used_model,
            }
            raw_data = OpenAIRawTrackedData(
                request_kwargs=self._request_kwargs,
                request_response=mocked_request_response,
                api_type=self._api_type,
                api_key=openai.api_key,
                api_provider=openai.api_type,
                timestamp=self._timestamp,
                timestamp_end=timestamp_end,
                organization=openai.organization,
            )
            queue_object = OpenAIQueueObject(raw_data)
            self._nebuly_queue.put(item=queue_object, timeout=0)

        return wrapped_generator()

    @staticmethod
    @abstractmethod
    def _track_generator_element(
        element: Any,
        output_text: str,
        timestamp_openai: Optional[int],
        used_model: Optional[str],
    ) -> Tuple[str, int, str]:
        ...


class TextCompletionStrategy(GeneratorWrappingStrategy):
    @staticmethod
    def _track_generator_element(
        element: Any,
        output_text: str,
        timestamp_openai: Optional[int],
        used_model: Optional[str],
    ) -> Tuple[str, int, str]:
        try:
            text = element["choices"][0]["text"]
            output_text += text
            if timestamp_openai is None:
                timestamp_openai = int(element["created"])
            if used_model is None:
                used_model = element["model"]
        except (KeyError, IndexError):
            pass
        return output_text, timestamp_openai, used_model


class ChatCompletionStrategy(GeneratorWrappingStrategy):
    @staticmethod
    def _track_generator_element(
        element: Any,
        output_text: str,
        timestamp_openai: Optional[int],
        used_model: Optional[str],
    ) -> Tuple[str, int, str]:
        try:
            delta = element["choices"][0]["delta"]
            if timestamp_openai is None:
                timestamp_openai = int(element["created"])
            if used_model is None:
                used_model = element["model"]
        except (KeyError, IndexError):
            return output_text, timestamp_openai, used_model
        try:
            text = delta["content"]
            output_text += text
        except KeyError:
            pass

        return output_text, timestamp_openai, used_model


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
                    wrap_strategy = ChatCompletionStrategy()
                elif api_type == OpenAIAPIType.TEXT_COMPLETION:
                    wrap_strategy = TextCompletionStrategy()
        except KeyError:
            pass

        request_response = wrap_strategy.wrap(
            nebuly_queue=self._nebuly_queue,
            original_method=original_method,
            request_kwargs=request_kwargs,
            api_type=api_type,
        )
        return request_response
