import copy
import datetime as dt
import unittest
from datetime import datetime
from typing import Generator
from unittest.mock import MagicMock, patch

import tests.trackers.openai_data as test_data
from nebuly.core.schemas import (
    DevelopmentPhase,
    Provider,
    TagData,
    Task,
)
from nebuly.trackers.openai import (
    APICallWrappingStrategy,
    AudioAPIBodyFiller,
    ChatCompletionStrategy,
    FineTuneAPIBodyFiller,
    ImageAPIBodyFiller,
    ModerationAPIBodyFiller,
    OpenAIAPIType,
    OpenAIAttributes,
    OpenAIDataPackageConverter,
    OpenAIQueueObject,
    OpenAIRawTrackedData,
    OpenAITracker,
    TextAPIBodyFiller,
    TextCompletionStrategy,
)


class TestTextAPIBodyFiller(unittest.TestCase):
    def test_fill_body_with_request_data__is_filling_completion_requests(self):
        filler = TextAPIBodyFiller()

        # STANDARD API
        request_kwargs = test_data.text_completion.request_kwargs
        request_response = test_data.text_completion.request_response

        body = OpenAIAttributes(
            project="test_project",
            development_phase=DevelopmentPhase.EXPERIMENTATION,
            task=Task.TEXT_GENERATION,
            api_type=OpenAIAPIType.TEXT_COMPLETION,
            api_key="test_api_key",
            timestamp=datetime.fromtimestamp(111111, tz=dt.timezone.utc),
            timestamp_end=datetime.fromtimestamp(222222, tz=dt.timezone.utc),
        )
        filler.fill_body_with_request_data(
            body=body,
            request_kwargs=request_kwargs,
            request_response=request_response,
        )

        self.assertEqual(body.model, request_kwargs["model"])
        self.assertEqual(
            body.n_output_tokens, request_response["usage"]["completion_tokens"]
        )
        self.assertEqual(
            body.n_input_tokens, request_response["usage"]["prompt_tokens"]
        )
        self.assertEqual(
            body.timestamp_openai,
            datetime.fromtimestamp(request_response["created"], tz=dt.timezone.utc),
        )

        # STREAM API
        stream_request_kwargs = test_data.text_completion_stream.request_kwargs
        stream_request_response = test_data.text_completion_stream.request_response

        body = OpenAIAttributes(
            project="test_project",
            development_phase=DevelopmentPhase.EXPERIMENTATION,
            task=Task.TEXT_GENERATION,
            api_type=OpenAIAPIType.TEXT_COMPLETION,
            api_key="test_api_key",
            timestamp=datetime.fromtimestamp(111111, tz=dt.timezone.utc),
            timestamp_end=datetime.fromtimestamp(222222, tz=dt.timezone.utc),
        )

        filler.fill_body_with_request_data(
            body=body,
            request_kwargs=stream_request_kwargs,
            request_response=stream_request_response,
        )

        self.assertEqual(body.model, request_kwargs["model"])
        self.assertEqual(body.user, request_kwargs["user"])
        self.assertEqual(
            body.n_output_tokens, request_response["usage"]["completion_tokens"]
        )
        self.assertEqual(
            body.n_input_tokens, request_response["usage"]["prompt_tokens"]
        )

    def test_fill_body_with_request_data__no_data_provided(self):
        filler = TextAPIBodyFiller()

        # STANDARD API
        request_kwargs = {}
        request_response = {
            key: value
            for key, value in test_data.text_completion.request_response.items()
            if key != "model"
        }

        body = OpenAIAttributes(
            project="test_project",
            development_phase=DevelopmentPhase.EXPERIMENTATION,
            task=Task.TEXT_GENERATION,
            api_type=OpenAIAPIType.TEXT_COMPLETION,
            api_key="test_api_key",
            timestamp=datetime.fromtimestamp(111111, tz=dt.timezone.utc),
            timestamp_end=datetime.fromtimestamp(222222, tz=dt.timezone.utc),
        )
        filler.fill_body_with_request_data(
            body=body,
            request_kwargs=request_kwargs,
            request_response=request_response,
        )

        self.assertIsNone(body.model)
        self.assertIsNone(body.user)

        # STREAM API
        stream_request_kwargs = {}
        stream_request_response = {
            key: value
            for key, value in test_data.text_completion_stream.request_response.items()
            if key != "model"
        }

        body = OpenAIAttributes(
            project="test_project",
            development_phase=DevelopmentPhase.EXPERIMENTATION,
            task=Task.TEXT_GENERATION,
            api_type=OpenAIAPIType.TEXT_COMPLETION,
            api_key="test_api_key",
            timestamp=datetime.fromtimestamp(111111, tz=dt.timezone.utc),
            timestamp_end=datetime.fromtimestamp(222222, tz=dt.timezone.utc),
        )

        filler.fill_body_with_request_data(
            body=body,
            request_kwargs=stream_request_kwargs,
            request_response=stream_request_response,
        )

        self.assertIsNone(body.model)
        self.assertIsNone(body.user)

    def test_fill_body_with_request_data__is_filling_chat_request(self):
        filler = TextAPIBodyFiller()

        for model in [
            "gpt-3.5-turbo-0301",
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo",
            "gpt-4-0314",
            "gpt-4-0613",
            "gpt-4",
        ]:
            # STANDARD API
            request_kwargs = test_data.chat_completion.request_kwargs
            request_response = copy.deepcopy(test_data.chat_completion.request_response)

            request_kwargs["model"] = model

            if model == "gpt-3.5-turbo-0301":
                # gpt-3.5-turbo-0301 is the only model that has a different
                # number of tokens for prompt and completion
                request_response["usage"]["completion_tokens"] = 9
                request_response["usage"]["prompt_tokens"] = 10
                request_response["usage"]["total_tokens"] = 19

            body = OpenAIAttributes(
                project="test_project",
                development_phase=DevelopmentPhase.EXPERIMENTATION,
                task=Task.TEXT_GENERATION,
                api_type=OpenAIAPIType.CHAT,
                api_key="test_api_key",
                timestamp=datetime.fromtimestamp(111111, tz=dt.timezone.utc),
                timestamp_end=datetime.fromtimestamp(222222, tz=dt.timezone.utc),
            )

            filler.fill_body_with_request_data(
                body=body,
                request_kwargs=request_kwargs,
                request_response=request_response,
            )

            self.assertEqual(body.model, request_kwargs["model"])
            self.assertEqual(body.user, request_kwargs["user"])
            self.assertEqual(
                body.n_output_tokens, request_response["usage"]["completion_tokens"]
            )
            self.assertEqual(
                body.n_input_tokens, request_response["usage"]["prompt_tokens"]
            )
            self.assertEqual(
                body.timestamp_openai,
                datetime.fromtimestamp(request_response["created"], tz=dt.timezone.utc),
            )

            # STREAM API
            stream_request_kwargs = test_data.chat_completion_stream.request_kwargs
            stream_request_response = test_data.chat_completion_stream.request_response

            stream_request_kwargs["model"] = model

            body = OpenAIAttributes(
                project="test_project",
                development_phase=DevelopmentPhase.EXPERIMENTATION,
                task=Task.CHAT,
                api_type=OpenAIAPIType.CHAT,
                api_key="test_api_key",
                timestamp=datetime.fromtimestamp(111111, tz=dt.timezone.utc),
                timestamp_end=datetime.fromtimestamp(222222, tz=dt.timezone.utc),
            )

            filler.fill_body_with_request_data(
                body=body,
                request_kwargs=stream_request_kwargs,
                request_response=stream_request_response,
            )

            self.assertEqual(body.model, request_kwargs["model"])
            self.assertEqual(body.user, request_kwargs["user"])
            self.assertEqual(
                body.n_output_tokens, request_response["usage"]["completion_tokens"]
            )
            self.assertEqual(
                body.n_input_tokens, request_response["usage"]["prompt_tokens"]
            )

    def test_fill_body_with_request_data__is_filling_edit_request(self):
        filler = TextAPIBodyFiller()

        request_kwargs = test_data.edit.request_kwargs
        request_response = test_data.edit.request_response

        body = OpenAIAttributes(
            project="test_project",
            development_phase=DevelopmentPhase.EXPERIMENTATION,
            task=Task.TEXT_EDITING,
            api_type=OpenAIAPIType.EDIT,
            api_key="test_api_key",
            timestamp=datetime.fromtimestamp(111111, tz=dt.timezone.utc),
            timestamp_end=datetime.fromtimestamp(222222, tz=dt.timezone.utc),
        )
        filler.fill_body_with_request_data(
            body=body,
            request_kwargs=request_kwargs,
            request_response=request_response,
        )

        self.assertEqual(body.model, request_kwargs["model"])
        self.assertEqual(body.user, request_kwargs["user"])
        self.assertEqual(
            body.n_output_tokens, request_response["usage"]["completion_tokens"]
        )
        self.assertEqual(
            body.n_input_tokens, request_response["usage"]["prompt_tokens"]
        )
        self.assertEqual(
            body.timestamp_openai,
            datetime.fromtimestamp(request_response["created"], tz=dt.timezone.utc),
        )

    def test_fill_body_with_request_data__is_filling_embedding_request(self):
        filler = TextAPIBodyFiller()

        request_kwargs = test_data.embedding.request_kwargs
        request_response = test_data.embedding.request_response

        body = OpenAIAttributes(
            project="test_project",
            development_phase=DevelopmentPhase.EXPERIMENTATION,
            task=Task.TEXT_EMBEDDING,
            api_type=OpenAIAPIType.EMBEDDING,
            api_key="test_api_key",
            timestamp=datetime.fromtimestamp(111111, tz=dt.timezone.utc),
            timestamp_end=datetime.fromtimestamp(222222, tz=dt.timezone.utc),
        )
        filler.fill_body_with_request_data(
            body=body,
            request_kwargs=request_kwargs,
            request_response=request_response,
        )

        self.assertEqual(body.model, request_kwargs["model"])
        self.assertEqual(body.user, request_kwargs["user"])
        self.assertEqual(
            body.n_input_tokens, request_response["usage"]["prompt_tokens"]
        )


class TestImageAPIBodyFiller(unittest.TestCase):
    def test_is_filling_body_with_request_data__is_filling_image_request(self):
        filler = ImageAPIBodyFiller()

        request_kwargs = test_data.image.request_kwargs
        request_response = test_data.image.request_response

        body = OpenAIAttributes(
            project="test_project",
            development_phase=DevelopmentPhase.EXPERIMENTATION,
            task=Task.IMAGE_GENERATION,
            api_type=OpenAIAPIType.IMAGE_CREATE,
            api_key="test_api_key",
            timestamp=datetime.fromtimestamp(111111, tz=dt.timezone.utc),
            timestamp_end=datetime.fromtimestamp(222222, tz=dt.timezone.utc),
        )
        filler.fill_body_with_request_data(
            body=body,
            request_kwargs=request_kwargs,
            request_response=request_response,
        )

        self.assertEqual(body.model, "dall-e")
        self.assertEqual(body.user, request_kwargs["user"])
        self.assertEqual(
            body.timestamp_openai,
            datetime.fromtimestamp(request_response["created"], tz=dt.timezone.utc),
        )
        self.assertEqual(body.n_output_images, request_kwargs["n"])
        self.assertEqual(body.image_size, request_kwargs["size"])

    def test_is_filling_body_with_request_data__no_data_provided(self):
        filler = ImageAPIBodyFiller()

        request_kwargs = {}
        request_response = test_data.image.request_response

        body = OpenAIAttributes(
            project="test_project",
            development_phase=DevelopmentPhase.EXPERIMENTATION,
            task=Task.IMAGE_GENERATION,
            api_type=OpenAIAPIType.IMAGE_CREATE,
            api_key="test_api_key",
            timestamp=datetime.fromtimestamp(111111, tz=dt.timezone.utc),
            timestamp_end=datetime.fromtimestamp(222222, tz=dt.timezone.utc),
        )

        filler.fill_body_with_request_data(
            body=body,
            request_kwargs=request_kwargs,
            request_response=request_response,
        )

        self.assertEqual(body.model, "dall-e")
        self.assertIsNone(body.user)


class TestAudioAPIBodyFiller(unittest.TestCase):
    @patch("nebuly.trackers.openai.get_media_file_length_in_seconds")
    def test_fill_body_with_request_data__is_filling_audio_request(self, mocked_method):
        filler = AudioAPIBodyFiller()
        mocked_method.return_value = 10

        request_kwargs = test_data.audio.request_kwargs
        request_response = test_data.audio.request_response

        body = OpenAIAttributes(
            project="test_project",
            development_phase=DevelopmentPhase.EXPERIMENTATION,
            task=Task.AUDIO_TRANSLATION,
            api_type=OpenAIAPIType.AUDIO_TRANSLATE,
            api_key="test_api_key",
            timestamp=datetime.fromtimestamp(111111, tz=dt.timezone.utc),
            timestamp_end=datetime.fromtimestamp(222222, tz=dt.timezone.utc),
        )
        filler.fill_body_with_request_data(
            body=body,
            request_kwargs=request_kwargs,
            request_response=request_response,
        )

        self.assertEqual(body.model, request_kwargs["model"])
        self.assertEqual(body.user, request_kwargs["user"])
        self.assertEqual(body.audio_duration_seconds, 10)

    @patch("nebuly.trackers.openai.get_media_file_length_in_seconds")
    def test_fill_body_with_request_data__no_data_provided(self, mocked_method):
        filler = AudioAPIBodyFiller()
        mocked_method.return_value = 10

        request_kwargs = {}
        request_response = test_data.audio.request_response

        body = OpenAIAttributes(
            project="test_project",
            development_phase=DevelopmentPhase.EXPERIMENTATION,
            task=Task.AUDIO_TRANSLATION,
            api_type=OpenAIAPIType.AUDIO_TRANSLATE,
            api_key="test_api_key",
            timestamp=datetime.fromtimestamp(111111, tz=dt.timezone.utc),
            timestamp_end=datetime.fromtimestamp(222222, tz=dt.timezone.utc),
        )
        filler.fill_body_with_request_data(
            body=body,
            request_kwargs=request_kwargs,
            request_response=request_response,
        )

        self.assertIsNone(body.model)
        self.assertIsNone(body.user)


class TestFineTuneBodyFiller(unittest.TestCase):
    def test_fill_body_with_request_data__is_filling_finetune_request(self):
        filler = FineTuneAPIBodyFiller()

        request_kwargs = test_data.finetune.request_kwargs
        request_response = test_data.finetune.request_response

        body = OpenAIAttributes(
            project="test_project",
            development_phase=DevelopmentPhase.EXPERIMENTATION,
            task=Task.FINETUNING,
            api_type=OpenAIAPIType.FINETUNE,
            api_key="test_api_key",
            timestamp=datetime.fromtimestamp(111111, tz=dt.timezone.utc),
            timestamp_end=datetime.fromtimestamp(222222, tz=dt.timezone.utc),
        )
        filler.fill_body_with_request_data(
            body=body,
            request_kwargs=request_kwargs,
            request_response=request_response,
        )

        self.assertEqual(body.model, request_response["model"])
        self.assertEqual(body.user, request_kwargs["user"])
        self.assertEqual(body.training_id, request_response["id"])
        self.assertEqual(body.training_file_id, request_kwargs["training_file"])
        self.assertEqual(
            body.timestamp_openai,
            datetime.fromtimestamp(request_response["created_at"], tz=dt.timezone.utc),
        )
        self.assertEqual(body.n_epochs, request_response["hyperparams"]["n_epochs"])

    def test_fill_body_with_request_data__no_data_provided(self):
        filler = FineTuneAPIBodyFiller()

        request_kwargs = {}
        request_response = {}

        body = OpenAIAttributes(
            project="test_project",
            development_phase=DevelopmentPhase.EXPERIMENTATION,
            task=Task.FINETUNING,
            api_type=OpenAIAPIType.FINETUNE,
            api_key="test_api_key",
            timestamp=datetime.fromtimestamp(111111, tz=dt.timezone.utc),
            timestamp_end=datetime.fromtimestamp(222222, tz=dt.timezone.utc),
        )
        filler.fill_body_with_request_data(
            body=body,
            request_kwargs=request_kwargs,
            request_response=request_response,
        )

        self.assertIsNone(body.model)
        self.assertIsNone(body.user)


class TestModerationAPIBodyFiller(unittest.TestCase):
    def test_fill_body_with_request_data__is_filling_moderation_request(self):
        filler = ModerationAPIBodyFiller()

        request_kwargs = test_data.finetune.request_kwargs
        request_response = test_data.finetune.request_response

        body = OpenAIAttributes(
            project="test_project",
            development_phase=DevelopmentPhase.EXPERIMENTATION,
            task=Task.TEXT_MODERATION,
            api_type=OpenAIAPIType.MODERATION,
            api_key="test_api_key",
            timestamp=datetime.fromtimestamp(111111, tz=dt.timezone.utc),
            timestamp_end=datetime.fromtimestamp(222222, tz=dt.timezone.utc),
        )
        filler.fill_body_with_request_data(
            body=body,
            request_kwargs=request_kwargs,
            request_response=request_response,
        )

        self.assertEqual(body.model, request_response["model"])
        self.assertEqual(body.user, request_kwargs["user"])

    def test_fill_body_with_request_data__no_data_provided(self):
        filler = ModerationAPIBodyFiller()

        request_kwargs = {}
        request_response = {}

        body = OpenAIAttributes(
            project="test_project",
            development_phase=DevelopmentPhase.EXPERIMENTATION,
            task=Task.TEXT_MODERATION,
            api_type=OpenAIAPIType.MODERATION,
            api_key="test_api_key",
            timestamp=datetime.fromtimestamp(111111, tz=dt.timezone.utc),
            timestamp_end=datetime.fromtimestamp(222222, tz=dt.timezone.utc),
        )
        filler.fill_body_with_request_data(
            body=body,
            request_kwargs=request_kwargs,
            request_response=request_response,
        )

        self.assertIsNone(body.model)
        self.assertIsNone(body.user)


class TestOpenAIDataPackageConverter(unittest.TestCase):
    def test_get_data_package__is_detecting_the_provider(self):
        raw_data = OpenAIRawTrackedData(
            timestamp=1010101,
            timestamp_end=20202020,
            request_kwargs=test_data.text_completion.request_kwargs,
            request_response=test_data.text_completion.request_response,
            api_type=OpenAIAPIType.TEXT_COMPLETION,
            api_key="test-key",
            api_provider="open_ai",
            organization="test-org",
        )
        tag_data = TagData()
        converter = OpenAIDataPackageConverter()
        data_package = converter.get_data_package(raw_data, tag_data)
        self.assertEqual(data_package.provider, Provider.OPENAI)

        raw_data.api_provider = "azure"
        data_package = converter.get_data_package(raw_data, tag_data)
        self.assertEqual(data_package.provider, Provider.AZURE_OPENAI)

    def test_get_data_package__is_detecting_the_task(self):
        raw_data = OpenAIRawTrackedData(
            timestamp=1010101,
            timestamp_end=20202020,
            request_kwargs=test_data.text_completion.request_kwargs,
            request_response=test_data.text_completion.request_response,
            api_type=OpenAIAPIType.TEXT_COMPLETION,
            api_key="test-key",
            api_provider="open_ai",
            organization="test-org",
        )
        tag_data = TagData()
        converter = OpenAIDataPackageConverter()
        data_package = converter.get_data_package(raw_data, tag_data)
        self.assertEqual(data_package.body.task, Task.TEXT_GENERATION)

        raw_data.api_type = OpenAIAPIType.CHAT
        data_package = converter.get_data_package(raw_data, tag_data)
        self.assertEqual(data_package.body.task, Task.CHAT)

        raw_data.api_type = OpenAIAPIType.EDIT
        data_package = converter.get_data_package(raw_data, tag_data)
        self.assertEqual(data_package.body.task, Task.TEXT_EDITING)

        raw_data.api_type = OpenAIAPIType.EMBEDDING
        data_package = converter.get_data_package(raw_data, tag_data)
        self.assertEqual(data_package.body.task, Task.TEXT_EMBEDDING)

        raw_data.api_type = OpenAIAPIType.IMAGE_CREATE
        data_package = converter.get_data_package(raw_data, tag_data)
        self.assertEqual(data_package.body.task, Task.IMAGE_GENERATION)

        raw_data.api_type = OpenAIAPIType.IMAGE_EDIT
        data_package = converter.get_data_package(raw_data, tag_data)
        self.assertEqual(data_package.body.task, Task.IMAGE_EDITING)

        raw_data.api_type = OpenAIAPIType.IMAGE_VARIATION
        data_package = converter.get_data_package(raw_data, tag_data)
        self.assertEqual(data_package.body.task, Task.IMAGE_VARIATION)

        raw_data.api_type = OpenAIAPIType.AUDIO_TRANSLATE
        data_package = converter.get_data_package(raw_data, tag_data)
        self.assertEqual(data_package.body.task, Task.AUDIO_TRANSLATION)

        raw_data.api_type = OpenAIAPIType.AUDIO_TRANSCRIBE
        data_package = converter.get_data_package(raw_data, tag_data)
        self.assertEqual(data_package.body.task, Task.AUDIO_TRANSCRIPTION)

        raw_data.api_type = OpenAIAPIType.FINETUNE
        data_package = converter.get_data_package(raw_data, tag_data)
        self.assertEqual(data_package.body.task, Task.FINETUNING)

        raw_data.api_type = OpenAIAPIType.MODERATION
        data_package = converter.get_data_package(raw_data, tag_data)
        self.assertEqual(data_package.body.task, Task.TEXT_MODERATION)

    def test_get_data_package__is_filling_the_request_independent_body_field(self):
        raw_data = OpenAIRawTrackedData(
            timestamp=1010101,
            timestamp_end=20202020,
            request_kwargs=test_data.text_completion.request_kwargs,
            request_response=test_data.text_completion.request_response,
            api_type=OpenAIAPIType.TEXT_COMPLETION,
            api_key="test-key",
            api_provider="open_ai",
            organization="test-org",
        )
        tag_data = TagData(
            project="test_project",
            development_phase=DevelopmentPhase.EXPERIMENTATION,
            task=Task.TEXT_GENERATION,
        )
        converter = OpenAIDataPackageConverter()
        data_package = converter.get_data_package(raw_data, tag_data)

        self.assertEqual(data_package.body.project, tag_data.project)
        self.assertEqual(
            data_package.body.development_phase, tag_data.development_phase
        )
        self.assertEqual(data_package.body.task, tag_data.task)
        self.assertEqual(data_package.body.api_type, raw_data.api_type)
        self.assertEqual(data_package.body.api_key, raw_data.api_key)
        self.assertEqual(
            data_package.body.timestamp,
            datetime.fromtimestamp(raw_data.timestamp, tz=dt.timezone.utc),
        )
        self.assertEqual(
            data_package.body.timestamp_end,
            datetime.fromtimestamp(raw_data.timestamp_end, tz=dt.timezone.utc),
        )


class TestOpenAIQueueObject(unittest.TestCase):
    def test_as_data_package__is_converting_the_raw_data(self):
        raw_data = OpenAIRawTrackedData(
            timestamp=1010101,
            timestamp_end=20202020,
            request_kwargs=test_data.text_completion.request_kwargs,
            request_response=test_data.text_completion.request_response,
            api_type=OpenAIAPIType.TEXT_COMPLETION,
            api_key="test-key",
            api_provider="open_ai",
            organization="test-org",
        )
        tag_data = TagData(
            project="test_project",
            development_phase=DevelopmentPhase.EXPERIMENTATION,
            task=Task.TEXT_GENERATION,
        )
        queue_object = OpenAIQueueObject(raw_data)
        queue_object.tag(tag_data)
        data_package = queue_object.as_data_package()

        self.assertEqual(data_package.provider, Provider.OPENAI)
        self.assertEqual(data_package.body.api_type, raw_data.api_type)
        self.assertEqual(data_package.body.api_key, raw_data.api_key)
        self.assertEqual(
            data_package.body.timestamp,
            datetime.fromtimestamp(raw_data.timestamp, tz=dt.timezone.utc),
        )
        self.assertEqual(
            data_package.body.timestamp_end,
            datetime.fromtimestamp(raw_data.timestamp_end, tz=dt.timezone.utc),
        )
        self.assertEqual(data_package.body.project, tag_data.project)
        self.assertEqual(
            data_package.body.development_phase, tag_data.development_phase
        )
        self.assertEqual(data_package.body.task, tag_data.task)
        self.assertEqual(data_package.body.model, raw_data.request_kwargs["model"])
        self.assertEqual(
            data_package.body.n_input_tokens,
            raw_data.request_response["usage"]["prompt_tokens"],
        )
        self.assertEqual(
            data_package.body.n_output_tokens,
            raw_data.request_response["usage"]["completion_tokens"],
        )
        self.assertEqual(
            data_package.body.timestamp_openai,
            datetime.fromtimestamp(
                raw_data.request_response["created"], tz=dt.timezone.utc
            ),
        )
        self.assertEqual(data_package.body.organization, raw_data.organization)


class TestAPICallWrappingStrategy(unittest.TestCase):
    mocked_nebuly_queue = None
    mocked_openai_method = None
    mocked_request_kwargs = None
    mocked_api_type = None
    mocked_value = 10

    def setUp(self) -> None:
        self.mocked_nebuly_queue = MagicMock()
        openai = MagicMock()
        openai.mock_method.return_value = self.mocked_value
        self.mocked_openai_method = openai.mock_method
        self.request_kwargs = test_data.text_completion.request_kwargs
        self.api_type = OpenAIAPIType.TEXT_COMPLETION

    def test_wrap__is_returning_the_response(self):
        strategy = APICallWrappingStrategy()
        response = strategy.wrap(
            nebuly_queue=self.mocked_nebuly_queue,
            original_method=self.mocked_openai_method,
            request_kwargs=self.request_kwargs,
            api_type=self.api_type,
        )
        self.assertEqual(response, self.mocked_value)

    def test_wrap__is_adding_the_data_to_the_queue(self):
        strategy = APICallWrappingStrategy()
        strategy.wrap(
            nebuly_queue=self.mocked_nebuly_queue,
            original_method=self.mocked_openai_method,
            request_kwargs=self.request_kwargs,
            api_type=self.api_type,
        )
        self.mocked_nebuly_queue.put.assert_called_once()


class TestChatCompletionGeneratorWrappingStrategy(unittest.TestCase):
    mocked_nebuly_queue = None
    mocked_openai_method = None
    mocked_request_kwargs = None
    mocked_api_type = None
    mocked_value = 10

    def setUp(self) -> None:
        self.mocked_nebuly_queue = MagicMock()
        openai = MagicMock()

        def generator_method():
            for _ in range(2):
                item = test_data.chat_completion_generator_response
                yield item

        openai.mock_method.return_value = generator_method()
        self.mocked_openai_method = openai.mock_method
        self.request_kwargs = test_data.text_completion.request_kwargs
        self.api_type = OpenAIAPIType.TEXT_COMPLETION

    def test_wrap__is_returning_a_generator(self):
        strategy = ChatCompletionStrategy()
        response = strategy.wrap(
            nebuly_queue=self.mocked_nebuly_queue,
            original_method=self.mocked_openai_method,
            request_kwargs=self.request_kwargs,
            api_type=self.api_type,
        )
        self.assertIsInstance(response, Generator)

    def test_wrap__is_adding_the_data_to_the_queue_when_generator_ends(self):
        strategy = TextCompletionStrategy()
        generator = strategy.wrap(
            nebuly_queue=self.mocked_nebuly_queue,
            original_method=self.mocked_openai_method,
            request_kwargs=self.request_kwargs,
            api_type=self.api_type,
        )
        for _ in generator:
            pass
        self.mocked_nebuly_queue.put.assert_called_once()


class TestTextCompletionGeneratorWrappingStrategy(unittest.TestCase):
    mocked_nebuly_queue = None
    mocked_openai_method = None
    mocked_request_kwargs = None
    mocked_api_type = None
    mocked_value = 10

    def setUp(self) -> None:
        self.mocked_nebuly_queue = MagicMock()
        openai = MagicMock()

        def generator_method():
            for _ in range(10):
                item = test_data.text_completion_generator_response
                yield item

        openai.mock_method.return_value = generator_method()
        self.mocked_openai_method = openai.mock_method
        self.request_kwargs = test_data.text_completion.request_kwargs
        self.api_type = OpenAIAPIType.TEXT_COMPLETION

    def test_wrap__is_returning_a_generator(self):
        strategy = TextCompletionStrategy()
        response = strategy.wrap(
            nebuly_queue=self.mocked_nebuly_queue,
            original_method=self.mocked_openai_method,
            request_kwargs=self.request_kwargs,
            api_type=self.api_type,
        )
        self.assertIsInstance(response, Generator)

    def test_wrap__is_adding_the_data_to_the_queue_when_generator_ends(self):
        strategy = TextCompletionStrategy()
        generator = strategy.wrap(
            nebuly_queue=self.mocked_nebuly_queue,
            original_method=self.mocked_openai_method,
            request_kwargs=self.request_kwargs,
            api_type=self.api_type,
        )
        for _ in generator:
            pass
        self.mocked_nebuly_queue.put.assert_called_once()


class TestOpenAITracker(unittest.TestCase):
    mocked_openai_response = "OpenAIResponse"
    mocked_kwargs = {"arg": "arg"}

    @patch("nebuly.trackers.openai.openai")
    def test_replace_sdk_functions__is_issuing_the_track_request_for_text_completion(
        self,
        mocked_openai: MagicMock,
    ) -> None:
        mocked_openai.api_type = "open_ai"
        mocked_nebuly_queue = MagicMock()
        mocked_openai.Completion.create.return_value = self.mocked_openai_response

        openai_tracker = OpenAITracker(nebuly_queue=mocked_nebuly_queue)
        openai_tracker.replace_sdk_functions()
        mocked_openai.Completion.create(**self.mocked_kwargs)

        mocked_nebuly_queue.put.assert_called_once()
        _, kwargs = mocked_nebuly_queue.put.call_args
        queue_object = kwargs["item"]
        self.assertEqual(
            first=queue_object._raw_data.request_kwargs, second=self.mocked_kwargs
        )
        self.assertEqual(
            first=queue_object._raw_data.request_response,
            second=self.mocked_openai_response,
        )
        self.assertEqual(
            first=queue_object._raw_data.api_type,
            second=OpenAIAPIType.TEXT_COMPLETION,
        )

    @patch("nebuly.trackers.openai.openai")
    def test_replace_sdk_functions__is_replacing_the_method_for_text_completion(
        self,
        mocked_openai: MagicMock,
    ) -> None:
        mocked_nebuly_queue = MagicMock()

        openai_tracker = OpenAITracker(nebuly_queue=mocked_nebuly_queue)
        openai_tracker.replace_sdk_functions()

        self.assertNotIsInstance(obj=mocked_openai.Completion.create, cls=MagicMock)

    @patch("nebuly.trackers.openai.openai")
    def test_replace_sdk_functions__is_issuing_the_track_request_for_chat_completion(
        self,
        mocked_openai: MagicMock,
    ):
        mocked_openai.api_type = "open_ai"
        mocked_nebuly_queue = MagicMock()
        mocked_openai.ChatCompletion.create.return_value = self.mocked_openai_response

        openai_tracker = OpenAITracker(nebuly_queue=mocked_nebuly_queue)
        openai_tracker.replace_sdk_functions()
        mocked_openai.ChatCompletion.create(**self.mocked_kwargs)

        mocked_nebuly_queue.put.assert_called_once()
        _, kwargs = mocked_nebuly_queue.put.call_args
        queue_object = kwargs["item"]
        self.assertEqual(
            first=queue_object._raw_data.request_kwargs, second=self.mocked_kwargs
        )
        self.assertEqual(
            first=queue_object._raw_data.request_response,
            second=self.mocked_openai_response,
        )
        self.assertEqual(
            first=queue_object._raw_data.api_type, second=OpenAIAPIType.CHAT
        )

    @patch("nebuly.trackers.openai.openai")
    def test_replace_sdk_functions__is_replacing_the_method_chat_completion(
        self,
        mocked_openai: MagicMock,
    ) -> None:
        mocked_nebuly_queue = MagicMock()

        openai_tracker = OpenAITracker(nebuly_queue=mocked_nebuly_queue)
        openai_tracker.replace_sdk_functions()

        self.assertNotIsInstance(obj=mocked_openai.ChatCompletion.create, cls=MagicMock)

    @patch("nebuly.trackers.openai.openai")
    def test_replace_sdk_functions__is_issuing_the_track_request_for_edit(
        self,
        mocked_openai: MagicMock,
    ) -> None:
        mocked_openai.api_type = "open_ai"
        mocked_nebuly_queue = MagicMock()
        mocked_openai.Edit.create.return_value = self.mocked_openai_response

        openai_tracker = OpenAITracker(nebuly_queue=mocked_nebuly_queue)
        openai_tracker.replace_sdk_functions()
        mocked_openai.Edit.create(**self.mocked_kwargs)

        mocked_nebuly_queue.put.assert_called_once()
        _, kwargs = mocked_nebuly_queue.put.call_args
        queue_object = kwargs["item"]
        self.assertEqual(
            first=queue_object._raw_data.request_kwargs, second=self.mocked_kwargs
        )
        self.assertEqual(
            first=queue_object._raw_data.request_response,
            second=self.mocked_openai_response,
        )
        self.assertEqual(
            first=queue_object._raw_data.api_type, second=OpenAIAPIType.EDIT
        )

    @patch("nebuly.trackers.openai.openai")
    def test_replace_sdk_functions__is_replacing_the_method_edit(
        self,
        mocked_openai: MagicMock,
    ) -> None:
        mocked_nebuly_queue = MagicMock()

        openai_tracker = OpenAITracker(nebuly_queue=mocked_nebuly_queue)
        openai_tracker.replace_sdk_functions()

        self.assertNotIsInstance(obj=mocked_openai.Edit.create, cls=MagicMock)

    @patch("nebuly.trackers.openai.openai")
    def test_replace_sdk_functions__is_issuing_the_track_request_for_embedding(
        self,
        mocked_openai: MagicMock,
    ):
        mocked_openai.api_type = "open_ai"
        mocked_nebuly_queue = MagicMock()
        mocked_openai.Embedding.create.return_value = self.mocked_openai_response

        openai_tracker = OpenAITracker(nebuly_queue=mocked_nebuly_queue)
        openai_tracker.replace_sdk_functions()
        mocked_openai.Embedding.create(**self.mocked_kwargs)

        mocked_nebuly_queue.put.assert_called_once()
        _, kwargs = mocked_nebuly_queue.put.call_args
        queue_object = kwargs["item"]
        self.assertEqual(
            first=queue_object._raw_data.request_kwargs, second=self.mocked_kwargs
        )
        self.assertEqual(
            first=queue_object._raw_data.request_response,
            second=self.mocked_openai_response,
        )
        self.assertEqual(
            first=queue_object._raw_data.api_type, second=OpenAIAPIType.EMBEDDING
        )

    @patch("nebuly.trackers.openai.openai")
    def test_replace_sdk_functions__is_replacing_the_method_embedding(
        self,
        mocked_openai: MagicMock,
    ) -> None:
        mocked_nebuly_queue = MagicMock()

        openai_tracker = OpenAITracker(nebuly_queue=mocked_nebuly_queue)
        openai_tracker.replace_sdk_functions()

        self.assertNotIsInstance(obj=mocked_openai.Embedding.create, cls=MagicMock)

    @patch("nebuly.trackers.openai.openai")
    def test_replace_sdk_functions__is_issuing_the_track_request_for_audio_transcribe(
        self,
        mocked_openai: MagicMock,
    ) -> None:
        mocked_openai.api_type = "open_ai"
        mocked_nebuly_queue = MagicMock()
        mocked_openai.Audio.transcribe.return_value = self.mocked_openai_response

        openai_tracker = OpenAITracker(nebuly_queue=mocked_nebuly_queue)
        openai_tracker.replace_sdk_functions()
        mocked_openai.Audio.transcribe(**self.mocked_kwargs)

        mocked_nebuly_queue.put.assert_called_once()
        _, kwargs = mocked_nebuly_queue.put.call_args
        queue_object = kwargs["item"]
        self.assertEqual(
            first=queue_object._raw_data.request_kwargs, second=self.mocked_kwargs
        )
        self.assertEqual(
            first=queue_object._raw_data.request_response,
            second=self.mocked_openai_response,
        )
        self.assertEqual(
            first=queue_object._raw_data.api_type,
            second=OpenAIAPIType.AUDIO_TRANSCRIBE,
        )

    @patch("nebuly.trackers.openai.openai")
    def test_replace_sdk_functions__is_replacing_the_method_audio_transcribe(
        self,
        mocked_openai: MagicMock,
    ) -> None:
        mocked_nebuly_queue = MagicMock()

        openai_tracker = OpenAITracker(nebuly_queue=mocked_nebuly_queue)
        openai_tracker.replace_sdk_functions()

        self.assertNotIsInstance(obj=mocked_openai.Audio.transcribe, cls=MagicMock)

    @patch("nebuly.trackers.openai.openai")
    def test_replace_sdk_functions__is_issuing_the_track_request_for_audio_translate(
        self,
        mocked_openai: MagicMock,
    ) -> None:
        mocked_openai.api_type = "open_ai"
        mocked_nebuly_queue = MagicMock()
        mocked_openai.Audio.translate.return_value = self.mocked_openai_response

        openai_tracker = OpenAITracker(nebuly_queue=mocked_nebuly_queue)
        openai_tracker.replace_sdk_functions()
        mocked_openai.Audio.translate(**self.mocked_kwargs)

        mocked_nebuly_queue.put.assert_called_once()
        _, kwargs = mocked_nebuly_queue.put.call_args
        queue_object = kwargs["item"]
        self.assertEqual(
            first=queue_object._raw_data.request_kwargs, second=self.mocked_kwargs
        )
        self.assertEqual(
            first=queue_object._raw_data.request_response,
            second=self.mocked_openai_response,
        )
        self.assertEqual(
            first=queue_object._raw_data.api_type,
            second=OpenAIAPIType.AUDIO_TRANSLATE,
        )

    @patch("nebuly.trackers.openai.openai")
    def test_replace_sdk_functions__is_replacing_the_method_for_audio_translate(
        self,
        mocked_openai: MagicMock,
    ) -> None:
        mocked_nebuly_queue = MagicMock()

        openai_tracker = OpenAITracker(nebuly_queue=mocked_nebuly_queue)
        openai_tracker.replace_sdk_functions()

        self.assertNotIsInstance(obj=mocked_openai.Audio.translate, cls=MagicMock)

    @patch("nebuly.trackers.openai.openai")
    def test_replace_sdk_functions__is_issuing_the_track_request_for_image_create(
        self,
        mocked_openai: MagicMock,
    ) -> None:
        mocked_openai.api_type = "open_ai"
        mocked_nebuly_queue = MagicMock()
        mocked_openai.Image.create.return_value = self.mocked_openai_response

        openai_tracker = OpenAITracker(nebuly_queue=mocked_nebuly_queue)
        openai_tracker.replace_sdk_functions()
        mocked_openai.Image.create(**self.mocked_kwargs)

        mocked_nebuly_queue.put.assert_called_once()
        _, kwargs = mocked_nebuly_queue.put.call_args
        queue_object = kwargs["item"]
        self.assertEqual(
            first=queue_object._raw_data.request_kwargs, second=self.mocked_kwargs
        )
        self.assertEqual(
            first=queue_object._raw_data.request_response,
            second=self.mocked_openai_response,
        )
        self.assertEqual(
            first=queue_object._raw_data.api_type, second=OpenAIAPIType.IMAGE_CREATE
        )

    @patch("nebuly.trackers.openai.openai")
    def test_replace_sdk_functions__is_replacing_the_method_image_create(
        self,
        mocked_openai: MagicMock,
    ) -> None:
        mocked_nebuly_queue = MagicMock()

        openai_tracker = OpenAITracker(nebuly_queue=mocked_nebuly_queue)
        openai_tracker.replace_sdk_functions()

        self.assertNotIsInstance(obj=mocked_openai.Image.create, cls=MagicMock)

    @patch("nebuly.trackers.openai.openai")
    def test_replace_sdk_functions__is_issuing_the_track_request_for_image_edit(
        self,
        mocked_openai: MagicMock,
    ) -> None:
        mocked_openai.api_type = "open_ai"
        mocked_nebuly_queue = MagicMock()
        mocked_openai.Image.create_edit.return_value = self.mocked_openai_response

        openai_tracker = OpenAITracker(nebuly_queue=mocked_nebuly_queue)
        openai_tracker.replace_sdk_functions()
        mocked_openai.Image.create_edit(**self.mocked_kwargs)

        mocked_nebuly_queue.put.assert_called_once()
        _, kwargs = mocked_nebuly_queue.put.call_args
        queue_object = kwargs["item"]
        self.assertEqual(
            first=queue_object._raw_data.request_kwargs, second=self.mocked_kwargs
        )
        self.assertEqual(
            first=queue_object._raw_data.request_response,
            second=self.mocked_openai_response,
        )
        self.assertEqual(
            first=queue_object._raw_data.api_type, second=OpenAIAPIType.IMAGE_EDIT
        )

    @patch("nebuly.trackers.openai.openai")
    def test_replace_sdk_functions__is_replacing_the_method_image_edit(
        self,
        mocked_openai: MagicMock,
    ) -> None:
        mocked_nebuly_queue = MagicMock()

        openai_tracker = OpenAITracker(nebuly_queue=mocked_nebuly_queue)
        openai_tracker.replace_sdk_functions()

        self.assertNotIsInstance(obj=mocked_openai.Image.create_edit, cls=MagicMock)

    @patch("nebuly.trackers.openai.openai")
    def test_replace_sdk_functions__is_issuing_the_track_request_for_image_variation(
        self,
        mocked_openai: MagicMock,
    ):
        mocked_openai.api_type = "open_ai"
        mocked_nebuly_queue = MagicMock()
        mocked_openai.Image.create_variation.return_value = self.mocked_openai_response

        openai_tracker = OpenAITracker(nebuly_queue=mocked_nebuly_queue)
        openai_tracker.replace_sdk_functions()
        mocked_openai.Image.create_variation(**self.mocked_kwargs)

        mocked_nebuly_queue.put.assert_called_once()
        _, kwargs = mocked_nebuly_queue.put.call_args
        queue_object = kwargs["item"]
        self.assertEqual(
            first=queue_object._raw_data.request_kwargs, second=self.mocked_kwargs
        )
        self.assertEqual(
            first=queue_object._raw_data.request_response,
            second=self.mocked_openai_response,
        )
        self.assertEqual(
            first=queue_object._raw_data.api_type,
            second=OpenAIAPIType.IMAGE_VARIATION,
        )

    @patch("nebuly.trackers.openai.openai")
    def test_replace_sdk_functions__is_replacing_the_method_image_variation(
        self,
        mocked_openai: MagicMock,
    ) -> None:
        mocked_nebuly_queue = MagicMock()

        openai_tracker = OpenAITracker(nebuly_queue=mocked_nebuly_queue)
        openai_tracker.replace_sdk_functions()

        self.assertNotIsInstance(
            obj=mocked_openai.Image.create_variation, cls=MagicMock
        )

    @patch("nebuly.trackers.openai.openai")
    def test_replace_sdk_functions__is_issuing_the_track_request_for_finetune(
        self,
        mocked_openai: MagicMock,
    ) -> None:
        mocked_openai.api_type = "open_ai"
        mocked_nebuly_queue = MagicMock()
        mocked_openai.FineTune.create.return_value = self.mocked_openai_response

        openai_tracker = OpenAITracker(nebuly_queue=mocked_nebuly_queue)
        openai_tracker.replace_sdk_functions()
        mocked_openai.FineTune.create(**self.mocked_kwargs)

        mocked_nebuly_queue.put.assert_called_once()
        _, kwargs = mocked_nebuly_queue.put.call_args
        queue_object = kwargs["item"]
        self.assertEqual(
            first=queue_object._raw_data.request_kwargs, second=self.mocked_kwargs
        )
        self.assertEqual(
            first=queue_object._raw_data.request_response,
            second=self.mocked_openai_response,
        )
        self.assertEqual(
            first=queue_object._raw_data.api_type, second=OpenAIAPIType.FINETUNE
        )

    @patch("nebuly.trackers.openai.openai")
    def test_replace_sdk_functions__is_replacing_the_method_finetune(
        self,
        mocked_openai: MagicMock,
    ) -> None:
        mocked_nebuly_queue = MagicMock()

        openai_tracker = OpenAITracker(nebuly_queue=mocked_nebuly_queue)
        openai_tracker.replace_sdk_functions()

        self.assertNotIsInstance(obj=mocked_openai.FineTune.create, cls=MagicMock)

    @patch("nebuly.trackers.openai.openai")
    def test_replace_sdk_functions__is_issuing_the_track_request_for_moderation(
        self,
        mocked_openai: MagicMock,
    ) -> None:
        mocked_openai.api_type = "open_ai"
        mocked_nebuly_queue = MagicMock()
        mocked_openai.Moderation.create.return_value = self.mocked_openai_response

        openai_tracker = OpenAITracker(nebuly_queue=mocked_nebuly_queue)
        openai_tracker.replace_sdk_functions()
        mocked_openai.Moderation.create(**self.mocked_kwargs)

        mocked_nebuly_queue.put.assert_called_once()
        _, kwargs = mocked_nebuly_queue.put.call_args
        queue_object = kwargs["item"]
        self.assertEqual(
            first=queue_object._raw_data.request_kwargs, second=self.mocked_kwargs
        )
        self.assertEqual(
            first=queue_object._raw_data.request_response,
            second=self.mocked_openai_response,
        )
        self.assertEqual(
            first=queue_object._raw_data.api_type, second=OpenAIAPIType.MODERATION
        )

    @patch("nebuly.trackers.openai.openai")
    def test_replace_sdk_functions__is_replacing_the_method_moderation(
        self,
        mocked_openai: MagicMock,
    ) -> None:
        mocked_nebuly_queue = MagicMock()

        openai_tracker = OpenAITracker(nebuly_queue=mocked_nebuly_queue)
        openai_tracker.replace_sdk_functions()

        self.assertNotIsInstance(obj=mocked_openai.Moderation.create, cls=MagicMock)
