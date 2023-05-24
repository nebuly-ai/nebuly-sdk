from unittest.mock import MagicMock, patch
import unittest

from nebuly.core.schemas import (
    NebulyDataPackage,
    DevelopmentPhase,
    Provider,
    TagData,
    Task,
)
from nebuly.trackers.openai import (
    OpenAIAPIType,
    OpenAIAttributes,
    OpenAIDataPackageConverter,
    OpenAIQueueObject,
    OpenAITracker,
)


class TestOpenAIDataPackageConverter(unittest.TestCase):
    mocked_timestamp = 1614807352
    mocked_timestamp_end = 1614807353
    mocked_api_key = "test_api_key"
    mocked_api_provider = "open_ai"
    text_request_kwargs = {
        "model": "text-davinci-003",
        "prompt": "Say this is a test",
        "max_tokens": 7,
        "temperature": 0,
        "top_p": 1,
        "n": 1,
        "stream": False,
        "logprobs": None,
        "stop": "\n",
    }
    text_request_response = {
        "id": "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7",
        "object": "text_completion",
        "created": 1589478378,
        "model": "text-davinci-003",
        "choices": [
            {
                "text": "\n\nThis is indeed a test",
                "index": 0,
                "logprobs": None,
                "finish_reason": "length",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
    }

    audio_file = MagicMock()
    audio_file.name = "french_audio.mp3"
    audio_request_kwargs = {"file": audio_file, "model": "whisper-1"}

    audio_request_response = {
        "text": (
            "Imagine the wildest idea that you've ever had, and you're "
            "curious about how it might scale to something that's a 100,"
            "a 1,000 times bigger."
        )
    }

    image_request_kwargs = {
        "prompt": "A cute baby sea otter",
        "n": 2,
        "size": "1024x1024",
    }

    image_request_response = {
        "created": 1589478378,
        "data": [{"url": "https://..."}, {"url": "https://..."}],
    }

    finetune_request_kwargs = {
        "training_file": "file-XGinujblHPwGLSztz8cPS8XY",
        "model": "curie",
    }
    finetune_request_response = {
        "id": "ft-AF1WoRqd3aJAHsqc9NY7iL8F",
        "object": "fine-tune",
        "model": "curie",
        "created_at": 1614807352,
        "events": [
            {
                "object": "fine-tune-event",
                "created_at": 1614807352,
                "level": "info",
                "message": (
                    "Job enqueued. Waiting for jobs ahead to complete. Queue number: 0."
                ),
            }
        ],
        "fine_tuned_model": None,
        "hyperparams": {
            "batch_size": 4,
            "learning_rate_multiplier": 0.1,
            "n_epochs": 4,
            "prompt_loss_weight": 0.1,
        },
        "organization_id": "org-...",
        "result_files": [],
        "status": "pending",
        "validation_files": [],
        "training_files": [
            {
                "id": "file-XGinujblHPwGLSztz8cPS8XY",
                "object": "file",
                "bytes": 1547276,
                "created_at": 1610062281,
                "filename": "my-data-train.jsonl",
                "purpose": "fine-tune-train",
            }
        ],
        "updated_at": 1614807352,
    }

    moderation_request_kwargs = {"input": "I want to kill them."}
    moderation_request_response = {
        "id": "modr-XXXXX",
        "model": "text-moderation-001",
        "results": [
            {
                "categories": {
                    "hate": False,
                    "hate/threatening": False,
                    "self-harm": False,
                    "sexual": False,
                    "sexual/minors": False,
                    "violence": False,
                    "violence/graphic": False,
                },
                "category_scores": {
                    "hate": 0.18805529177188873,
                    "hate/threatening": 0.0001250059431185946,
                    "self-harm": 0.0003706029092427343,
                    "sexual": 0.0008735615410842001,
                    "sexual/minors": 0.0007470346172340214,
                    "violence": 0.0041268812492489815,
                    "violence/graphic": 0.00023186142789199948,
                },
                "flagged": False,
            }
        ],
    }

    def test_get_data_package__is_returning_an_instance_of_nebuly_data_package(
        self,
    ) -> None:
        mocked_api_type = OpenAIAPIType.TEXT_COMPLETION.value
        tag_data = TagData()
        data_converter = OpenAIDataPackageConverter()
        request_data: NebulyDataPackage = data_converter.get_data_package(
            tag_data=tag_data,
            request_kwargs=self.text_request_kwargs,
            request_response=self.text_request_response,
            api_type=mocked_api_type,
            api_key=self.mocked_api_key,
            api_provider=self.mocked_api_provider,
            timestamp=self.mocked_timestamp,
            timestamp_end=self.mocked_timestamp_end,
        )

        self.assertIsInstance(obj=request_data, cls=NebulyDataPackage)

    def test_get_data_package__is_returning_the_right_api_provider(
        self,
    ) -> None:
        mocked_api_type = OpenAIAPIType.TEXT_COMPLETION.value
        tag_data = TagData()
        data_converter = OpenAIDataPackageConverter()
        request_data: NebulyDataPackage = data_converter.get_data_package(
            tag_data=tag_data,
            request_kwargs=self.text_request_kwargs,
            request_response=self.text_request_response,
            api_type=mocked_api_type,
            api_key=self.mocked_api_key,
            api_provider="open_ai",
            timestamp=self.mocked_timestamp,
            timestamp_end=self.mocked_timestamp_end,
        )

        self.assertEqual(first=request_data.kind, second=Provider.OPENAI)

        request_data: NebulyDataPackage = data_converter.get_data_package(
            tag_data=tag_data,
            request_kwargs=self.text_request_kwargs,
            request_response=self.text_request_response,
            api_type=mocked_api_type,
            api_key=self.mocked_api_key,
            api_provider="azure",
            timestamp=self.mocked_timestamp,
            timestamp_end=self.mocked_timestamp_end,
        )
        self.assertEqual(first=request_data.kind, second=Provider.AZURE_OPENAI)

    def test_as_data_package__is_returning_the_correct_data_for_text_api(
        self,
    ) -> None:
        api_type = OpenAIAPIType.TEXT_COMPLETION
        tag_data = TagData()
        data_converter = OpenAIDataPackageConverter()

        request_data = data_converter.get_data_package(
            tag_data=tag_data,
            request_kwargs=self.text_request_kwargs,
            request_response=self.text_request_response,
            api_type=api_type.value,
            api_key=self.mocked_api_key,
            api_provider=self.mocked_api_provider,
            timestamp=self.mocked_timestamp,
            timestamp_end=self.mocked_timestamp_end,
        )

        expected_body: OpenAIAttributes = OpenAIAttributes(
            project="unknown",
            phase=DevelopmentPhase.UNKNOWN,
            task=Task.TEXT_GENERATION,
            api_type=OpenAIAPIType.TEXT_COMPLETION.value,
            api_key=self.mocked_api_key,
            timestamp=self.mocked_timestamp,
            timestamp_end=self.mocked_timestamp_end,
            timestamp_openai=1589478378,
            model="text-davinci-003",
            n_prompt_tokens=5,
            n_output_tokens=7,
            n_output_images=None,
            image_size=None,
            audio_duration_seconds=None,
            training_file_id=None,
            training_id=None,
        )
        expected_response = NebulyDataPackage(
            kind=Provider.OPENAI,
            body=expected_body,
        )

        self.assertEqual(first=request_data, second=expected_response)

        api_type = OpenAIAPIType.CHAT
        tag_data.task = Task.UNKNOWN
        request_data = data_converter.get_data_package(
            tag_data=tag_data,
            request_kwargs=self.text_request_kwargs,
            request_response=self.text_request_response,
            api_type=api_type.value,
            api_key=self.mocked_api_key,
            api_provider=self.mocked_api_provider,
            timestamp=self.mocked_timestamp,
            timestamp_end=self.mocked_timestamp_end,
        )
        expected_response.body.task = Task.CHAT
        expected_response.body.api_type = OpenAIAPIType.CHAT.value
        self.assertEqual(first=request_data, second=expected_response)

        api_type = OpenAIAPIType.EMBEDDING
        tag_data.task = Task.UNKNOWN
        request_data = data_converter.get_data_package(
            tag_data=tag_data,
            request_kwargs=self.text_request_kwargs,
            request_response=self.text_request_response,
            api_type=api_type.value,
            api_key=self.mocked_api_key,
            api_provider=self.mocked_api_provider,
            timestamp=self.mocked_timestamp,
            timestamp_end=self.mocked_timestamp_end,
        )
        expected_response.body.task = Task.TEXT_EMBEDDING
        expected_response.body.api_type = OpenAIAPIType.EMBEDDING.value
        self.assertEqual(first=request_data, second=expected_response)

        api_type = OpenAIAPIType.EDIT
        tag_data.task = Task.UNKNOWN
        request_data: NebulyDataPackage = data_converter.get_data_package(
            tag_data=tag_data,
            request_kwargs=self.text_request_kwargs,
            request_response=self.text_request_response,
            api_type=api_type.value,
            api_key=self.mocked_api_key,
            api_provider=self.mocked_api_provider,
            timestamp=self.mocked_timestamp,
            timestamp_end=self.mocked_timestamp_end,
        )
        expected_response.body.task = Task.TEXT_EDITING
        expected_response.body.api_type = OpenAIAPIType.EDIT.value
        self.assertEqual(first=request_data, second=expected_response)

    @patch("nebuly.trackers.openai.openai")
    @patch("nebuly.trackers.openai.get_media_file_length_in_seconds")
    def test_get_data_package__is_returning_the_correct_data_for_audio_api(
        self,
        mocked_function: MagicMock,
        mocked_openai: MagicMock,
    ) -> None:
        mocked_openai.api_type = "open_ai"
        mocked_function.return_value = 10
        api_type = OpenAIAPIType.AUDIO_TRANSCRIBE
        tag_data = TagData()
        data_converter = OpenAIDataPackageConverter()

        request_data: NebulyDataPackage = data_converter.get_data_package(
            tag_data=tag_data,
            request_kwargs=self.audio_request_kwargs,
            request_response=self.audio_request_response,
            api_type=api_type.value,
            api_key=self.mocked_api_key,
            api_provider=self.mocked_api_provider,
            timestamp=self.mocked_timestamp,
            timestamp_end=self.mocked_timestamp_end,
        )

        expected_body: OpenAIAttributes = OpenAIAttributes(
            project="unknown",
            phase=DevelopmentPhase.UNKNOWN,
            task=Task.AUDIO_TRANSCRIPTION,
            api_type=OpenAIAPIType.AUDIO_TRANSCRIBE.value,
            api_key=self.mocked_api_key,
            timestamp=self.mocked_timestamp,
            timestamp_end=self.mocked_timestamp_end,
            timestamp_openai=None,
            model="whisper-1",
            n_prompt_tokens=None,
            n_output_tokens=None,
            n_output_images=None,
            image_size=None,
            audio_duration_seconds=10,
            training_file_id=None,
            training_id=None,
        )
        expected_response = NebulyDataPackage(
            kind=Provider.OPENAI,
            body=expected_body,
        )

        self.assertEqual(first=request_data, second=expected_response)

        api_type = OpenAIAPIType.AUDIO_TRANSLATE
        tag_data.task = Task.UNKNOWN
        request_data = data_converter.get_data_package(
            tag_data=tag_data,
            request_kwargs=self.audio_request_kwargs,
            request_response=self.audio_request_response,
            api_type=api_type.value,
            api_key=self.mocked_api_key,
            api_provider=self.mocked_api_provider,
            timestamp=self.mocked_timestamp,
            timestamp_end=self.mocked_timestamp_end,
        )
        expected_response.body.task = Task.AUDIO_TRANSLATION
        expected_response.body.api_type = OpenAIAPIType.AUDIO_TRANSLATE.value
        self.assertEqual(first=request_data, second=expected_response)

    @patch("nebuly.trackers.openai.openai")
    def test_get_data_package__is_returning_the_correct_data_for_image_api(
        self,
        mocked_openai: MagicMock,
    ) -> None:
        mocked_openai.api_type = "open_ai"
        api_type = OpenAIAPIType.IMAGE_CREATE
        tag_data = TagData()
        data_converter = OpenAIDataPackageConverter()

        request_data: NebulyDataPackage = data_converter.get_data_package(
            tag_data=tag_data,
            request_kwargs=self.image_request_kwargs,
            request_response=self.image_request_response,
            api_type=api_type.value,
            api_key=self.mocked_api_key,
            api_provider=self.mocked_api_provider,
            timestamp=self.mocked_timestamp,
            timestamp_end=self.mocked_timestamp_end,
        )

        expected_body: OpenAIAttributes = OpenAIAttributes(
            project="unknown",
            phase=DevelopmentPhase.UNKNOWN,
            task=Task.IMAGE_GENERATION,
            api_type=OpenAIAPIType.IMAGE_CREATE.value,
            api_key=self.mocked_api_key,
            timestamp=self.mocked_timestamp,
            timestamp_end=self.mocked_timestamp_end,
            timestamp_openai=1589478378,
            model="dall-e",
            n_prompt_tokens=None,
            n_output_tokens=None,
            n_output_images=2,
            image_size="1024x1024",
            audio_duration_seconds=None,
            training_file_id=None,
            training_id=None,
        )
        expected_response = NebulyDataPackage(
            kind=Provider.OPENAI,
            body=expected_body,
        )

        self.assertEqual(first=request_data, second=expected_response)

        api_type = OpenAIAPIType.IMAGE_EDIT
        tag_data.task = Task.UNKNOWN
        request_data = data_converter.get_data_package(
            tag_data=tag_data,
            request_kwargs=self.image_request_kwargs,
            request_response=self.image_request_response,
            api_type=api_type.value,
            api_key=self.mocked_api_key,
            api_provider=self.mocked_api_provider,
            timestamp=self.mocked_timestamp,
            timestamp_end=self.mocked_timestamp_end,
        )
        expected_response.body.task = Task.IMAGE_EDITING
        expected_response.body.api_type = OpenAIAPIType.IMAGE_EDIT.value
        self.assertEqual(first=request_data, second=expected_response)

        api_type = OpenAIAPIType.IMAGE_VARIATION
        tag_data.task = Task.UNKNOWN
        request_data = data_converter.get_data_package(
            tag_data=tag_data,
            request_kwargs=self.image_request_kwargs,
            request_response=self.image_request_response,
            api_type=api_type.value,
            api_key=self.mocked_api_key,
            api_provider=self.mocked_api_provider,
            timestamp=self.mocked_timestamp,
            timestamp_end=self.mocked_timestamp_end,
        )
        expected_response.body.task = Task.IMAGE_VARIATION
        expected_response.body.api_type = OpenAIAPIType.IMAGE_VARIATION.value
        self.assertEqual(first=request_data, second=expected_response)

    @patch("nebuly.trackers.openai.openai")
    def test_get_data_package__is_returning_the_correct_data_for_finetune_api(
        self,
        mocked_openai: MagicMock,
    ) -> None:
        mocked_openai.api_type = "open_ai"
        api_type = OpenAIAPIType.FINETUNE
        tag_data = TagData()
        data_converter = OpenAIDataPackageConverter()
        request_data: NebulyDataPackage = data_converter.get_data_package(
            tag_data=tag_data,
            request_kwargs=self.finetune_request_kwargs,
            request_response=self.finetune_request_response,
            api_type=api_type.value,
            api_key=self.mocked_api_key,
            api_provider=self.mocked_api_provider,
            timestamp=self.mocked_timestamp,
            timestamp_end=self.mocked_timestamp_end,
        )

        expected_body: OpenAIAttributes = OpenAIAttributes(
            project="unknown",
            phase=DevelopmentPhase.UNKNOWN,
            task=Task.FINETUNING,
            api_type=OpenAIAPIType.FINETUNE.value,
            api_key=self.mocked_api_key,
            timestamp=self.mocked_timestamp,
            timestamp_end=self.mocked_timestamp_end,
            timestamp_openai=1614807352,
            model="curie",
            n_prompt_tokens=None,
            n_output_tokens=None,
            n_output_images=None,
            image_size=None,
            audio_duration_seconds=None,
            training_file_id="file-XGinujblHPwGLSztz8cPS8XY",
            training_id="ft-AF1WoRqd3aJAHsqc9NY7iL8F",
        )
        expected_response = NebulyDataPackage(
            kind=Provider.OPENAI,
            body=expected_body,
        )
        print(request_data)
        print(expected_response)
        self.assertEqual(first=request_data, second=expected_response)

    @patch("nebuly.trackers.openai.openai")
    def test_get_data_package__is_returning_the_correct_data_for_moderation_api(
        self,
        mocked_openai: MagicMock,
    ) -> None:
        mocked_openai.api_type = "open_ai"
        api_type = OpenAIAPIType.MODERATION
        tag_data = TagData()
        data_converter = OpenAIDataPackageConverter()
        request_data: NebulyDataPackage = data_converter.get_data_package(
            tag_data=tag_data,
            request_kwargs=self.moderation_request_kwargs,
            request_response=self.moderation_request_response,
            api_type=api_type.value,
            api_key=self.mocked_api_key,
            api_provider=self.mocked_api_provider,
            timestamp=self.mocked_timestamp,
            timestamp_end=self.mocked_timestamp_end,
        )

        expected_body: OpenAIAttributes = OpenAIAttributes(
            project="unknown",
            phase=DevelopmentPhase.UNKNOWN,
            task=Task.TEXT_MODERATION,
            api_type=OpenAIAPIType.MODERATION.value,
            api_key=self.mocked_api_key,
            timestamp=self.mocked_timestamp,
            timestamp_end=self.mocked_timestamp_end,
            timestamp_openai=None,
            model="text-moderation-001",
            n_prompt_tokens=None,
            n_output_tokens=None,
            n_output_images=None,
            image_size=None,
            audio_duration_seconds=None,
            training_file_id=None,
            training_id=None,
        )

        expected_response = NebulyDataPackage(
            kind=Provider.OPENAI,
            body=expected_body,
        )

        self.assertEqual(first=request_data, second=expected_response)


class TestOpenAIQueueObject(unittest.TestCase):
    def test_as_data_package__is_calling_the_get_data_package_method_of_the_converter(
        self,
    ) -> None:
        mocked_converter = MagicMock()
        tag_data = TagData()
        mocked_converter.get_data_package.return_value = "data_package"
        mocked_openai_queue_object = OpenAIQueueObject(
            data_package_converter=mocked_converter,
            request_kwargs={"request_kwargs": "request_kwargs"},
            request_response={"request_response": "request_response"},
            api_type="api_type",
            api_key="test_api_key",
            api_provider="open_ai",
            timestamp=2.3,
            timestamp_end=2.4,
        )
        mocked_openai_queue_object.as_data_package()

        mocked_converter.get_data_package.assert_called_once_with(
            tag_data=tag_data,
            request_kwargs={"request_kwargs": "request_kwargs"},
            request_response={"request_response": "request_response"},
            api_type="api_type",
            api_key="test_api_key",
            api_provider="open_ai",
            timestamp=2.3,
            timestamp_end=2.4,
        )


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
        self.assertEqual(first=queue_object._request_kwargs, second=self.mocked_kwargs)
        self.assertEqual(
            first=queue_object._request_response,
            second=self.mocked_openai_response,
        )
        self.assertEqual(
            first=queue_object._api_type,
            second=OpenAIAPIType.TEXT_COMPLETION.value,
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
        self.assertEqual(first=queue_object._request_kwargs, second=self.mocked_kwargs)
        self.assertEqual(
            first=queue_object._request_response,
            second=self.mocked_openai_response,
        )
        self.assertEqual(first=queue_object._api_type, second=OpenAIAPIType.CHAT.value)

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
        self.assertEqual(first=queue_object._request_kwargs, second=self.mocked_kwargs)
        self.assertEqual(
            first=queue_object._request_response,
            second=self.mocked_openai_response,
        )
        self.assertEqual(first=queue_object._api_type, second=OpenAIAPIType.EDIT.value)

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
        self.assertEqual(first=queue_object._request_kwargs, second=self.mocked_kwargs)
        self.assertEqual(
            first=queue_object._request_response,
            second=self.mocked_openai_response,
        )
        self.assertEqual(
            first=queue_object._api_type, second=OpenAIAPIType.EMBEDDING.value
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
        self.assertEqual(first=queue_object._request_kwargs, second=self.mocked_kwargs)
        self.assertEqual(
            first=queue_object._request_response,
            second=self.mocked_openai_response,
        )
        self.assertEqual(
            first=queue_object._api_type,
            second=OpenAIAPIType.AUDIO_TRANSCRIBE.value,
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
        self.assertEqual(first=queue_object._request_kwargs, second=self.mocked_kwargs)
        self.assertEqual(
            first=queue_object._request_response,
            second=self.mocked_openai_response,
        )
        self.assertEqual(
            first=queue_object._api_type,
            second=OpenAIAPIType.AUDIO_TRANSLATE.value,
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
        self.assertEqual(first=queue_object._request_kwargs, second=self.mocked_kwargs)
        self.assertEqual(
            first=queue_object._request_response,
            second=self.mocked_openai_response,
        )
        self.assertEqual(
            first=queue_object._api_type, second=OpenAIAPIType.IMAGE_CREATE.value
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
        self.assertEqual(first=queue_object._request_kwargs, second=self.mocked_kwargs)
        self.assertEqual(
            first=queue_object._request_response,
            second=self.mocked_openai_response,
        )
        self.assertEqual(
            first=queue_object._api_type, second=OpenAIAPIType.IMAGE_EDIT.value
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
        self.assertEqual(first=queue_object._request_kwargs, second=self.mocked_kwargs)
        self.assertEqual(
            first=queue_object._request_response,
            second=self.mocked_openai_response,
        )
        self.assertEqual(
            first=queue_object._api_type,
            second=OpenAIAPIType.IMAGE_VARIATION.value,
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
        self.assertEqual(first=queue_object._request_kwargs, second=self.mocked_kwargs)
        self.assertEqual(
            first=queue_object._request_response,
            second=self.mocked_openai_response,
        )
        self.assertEqual(
            first=queue_object._api_type, second=OpenAIAPIType.FINETUNE.value
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
        self.assertEqual(first=queue_object._request_kwargs, second=self.mocked_kwargs)
        self.assertEqual(
            first=queue_object._request_response,
            second=self.mocked_openai_response,
        )
        self.assertEqual(
            first=queue_object._api_type, second=OpenAIAPIType.MODERATION.value
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
