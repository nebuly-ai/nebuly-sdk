import unittest
from unittest.mock import MagicMock, patch
import time

from nebuly.core.schemas import (
    NebulyDataPackage,
    DevelopmentPhase,
    Task,
    TagData,
    Provider,
)
from nebuly.trackers.openai import (
    OpenAITracker,
    OpenaiAIDataPackageConverter,
    OpenAIAPIType,
)


class TestOpenaiAIDataPackageConverter(unittest.TestCase):
    mocked_timestamp = 1614807352

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
    text_response = {
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

    audio_response = {
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

    image_response = {
        "created": 1589478378,
        "data": [{"url": "https://..."}, {"url": "https://..."}],
    }

    finetune_request_kwargs = {
        "training_file": "file-XGinujblHPwGLSztz8cPS8XY",
        "model": "curie",
    }
    finetune_response = {
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
    moderation_response = {
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

    @patch("nebuly.trackers.openai.openai")
    def test_init__is_detecting_the_openai_provider(
        self,
        mocked_openai,
    ):
        mocked_openai.api_type = "open_ai"
        data_converter = OpenaiAIDataPackageConverter(
            request_kwargs=MagicMock(),
            request_response=MagicMock(),
            api_type=MagicMock(),
        )
        self.assertEqual(data_converter._provider, Provider.OPENAI)
        mocked_openai.api_type = "azure"
        data_converter = OpenaiAIDataPackageConverter(
            request_kwargs=MagicMock(),
            request_response=MagicMock(),
            api_type=MagicMock(),
        )
        self.assertEqual(data_converter._provider, Provider.AZURE_OPENAI)

    @patch("nebuly.trackers.openai.openai")
    def test_init__is_assigning_the_timestamp(
        self,
        mocked_openai,
    ):
        mocked_openai.api_type = "open_ai"
        data_converter = OpenaiAIDataPackageConverter(
            request_kwargs=MagicMock(),
            request_response=MagicMock(),
            api_type=MagicMock(),
        )
        timestamp = time.time()
        self.assertAlmostEqual(data_converter._timestamp, timestamp, delta=1)

    @patch("nebuly.trackers.openai.openai")
    def test_get_data_package__is_returning_an_instance_of_nebuly_data_package(
        self,
        mocked_openai,
    ):
        mocked_openai.api_type = "open_ai"
        mocked_api_type = OpenAIAPIType.TEXT_COMPLETION
        tag_data = TagData(
            project="unknown_project",
            phase=DevelopmentPhase.UNKNOWN,
            task=Task.UNDETECTED,
        )
        data_converter = OpenaiAIDataPackageConverter(
            request_kwargs=self.text_request_kwargs,
            request_response=self.text_response,
            api_type=mocked_api_type,
        )
        data_converter.set_tag_data(tag_data)
        request_data = data_converter.get_data_package()

        self.assertIsInstance(request_data, NebulyDataPackage)

    @patch("nebuly.trackers.openai.openai")
    def test_as_data_package__is_returning_the_correct_data_for_text_api(
        self,
        mocked_openai,
    ):
        mocked_openai.api_type = "open_ai"
        mocked_api_type = OpenAIAPIType.TEXT_COMPLETION
        tag_data = TagData(
            project="unknown_project",
            phase=DevelopmentPhase.UNKNOWN,
            task=Task.UNDETECTED,
        )
        data_converter = OpenaiAIDataPackageConverter(
            request_kwargs=self.text_request_kwargs,
            request_response=self.text_response,
            api_type=mocked_api_type,
        )
        data_converter._timestamp = self.mocked_timestamp
        data_converter.set_tag_data(tag_data)
        request_data = data_converter.get_data_package()
        expected_response = NebulyDataPackage(
            project="unknown_project",
            phase="unknown",
            task=Task.TEXT_GENERATION,
            provider="openai",
            api_type=OpenAIAPIType.TEXT_COMPLETION.value,
            timestamp=self.mocked_timestamp,
            model="text-davinci-003",
            n_prompt_tokens=5,
            n_output_tokens=7,
            n_output_images=None,
            image_size=None,
            audio_duration_seconds=None,
        )
        self.assertEqual(request_data, expected_response)

        data_converter._api_type = OpenAIAPIType.CHAT
        data_converter._tag_data.task = Task.UNDETECTED
        request_data = data_converter.get_data_package()
        expected_response.task = Task.CHAT
        expected_response.api_type = OpenAIAPIType.CHAT.value
        self.assertEqual(request_data, expected_response)

        data_converter._api_type = OpenAIAPIType.EMBEDDING
        data_converter._tag_data.task = Task.UNDETECTED
        request_data = data_converter.get_data_package()
        expected_response.task = Task.TEXT_EMBEDDING
        expected_response.api_type = OpenAIAPIType.EMBEDDING.value
        self.assertEqual(request_data, expected_response)

        data_converter._api_type = OpenAIAPIType.EDIT
        data_converter._tag_data.task = Task.UNDETECTED
        request_data = data_converter.get_data_package()
        expected_response.task = Task.TEXT_EDITING
        expected_response.api_type = OpenAIAPIType.EDIT.value
        self.assertEqual(request_data, expected_response)

    @patch("nebuly.trackers.openai.openai")
    @patch("nebuly.trackers.openai.get_media_file_length_in_seconds")
    def test_get_data_package__is_returning_the_correct_data_for_audio_api(
        self,
        mocked_function,
        mocked_openai,
    ):
        mocked_openai.api_type = "open_ai"
        mocked_function.return_value = 10
        mocked_api_type = OpenAIAPIType.AUDIO_TRANSCRIBE
        tag_data = TagData(
            project="unknown_project",
            phase=DevelopmentPhase.UNKNOWN,
            task=Task.UNDETECTED,
        )
        data_converter = OpenaiAIDataPackageConverter(
            request_kwargs=self.audio_request_kwargs,
            request_response=self.audio_response,
            api_type=mocked_api_type,
        )
        data_converter._timestamp = self.mocked_timestamp
        data_converter.set_tag_data(tag_data)
        request_data = data_converter.get_data_package()

        expected_response = NebulyDataPackage(
            project="unknown_project",
            phase="unknown",
            task=Task.AUDIO_TRANSCRIPTION,
            provider="openai",
            api_type=OpenAIAPIType.AUDIO_TRANSCRIBE.value,
            timestamp=self.mocked_timestamp,
            model="whisper-1",
            n_prompt_tokens=None,
            n_output_tokens=None,
            n_output_images=None,
            image_size=None,
            audio_duration_seconds=10,
        )

        self.assertEqual(request_data, expected_response)

        data_converter._api_type = OpenAIAPIType.AUDIO_TRANSLATE
        data_converter._tag_data.task = Task.UNDETECTED
        request_data = data_converter.get_data_package()
        expected_response.task = Task.AUDIO_TRANSLATION
        expected_response.api_type = OpenAIAPIType.AUDIO_TRANSLATE.value
        self.assertEqual(request_data, expected_response)

    @patch("nebuly.trackers.openai.openai")
    def test_get_data_package__is_returning_the_correct_data_for_image_api(
        self,
        mocked_openai,
    ):
        mocked_openai.api_type = "open_ai"
        mocked_api_type = OpenAIAPIType.IMAGE_CREATE
        tag_data = TagData(
            project="unknown_project",
            phase=DevelopmentPhase.UNKNOWN,
            task=Task.UNDETECTED,
        )
        data_converter = OpenaiAIDataPackageConverter(
            request_kwargs=self.image_request_kwargs,
            request_response=self.image_response,
            api_type=mocked_api_type,
        )
        data_converter._timestamp = self.mocked_timestamp
        data_converter.set_tag_data(tag_data)
        request_data = data_converter.get_data_package()
        expected_response = NebulyDataPackage(
            project="unknown_project",
            phase="unknown",
            task=Task.IMAGE_GENERATION,
            provider="openai",
            api_type=OpenAIAPIType.IMAGE_CREATE.value,
            timestamp=self.mocked_timestamp,
            model=None,
            n_prompt_tokens=None,
            n_output_tokens=None,
            n_output_images=2,
            image_size="1024x1024",
            audio_duration_seconds=None,
        )

        print(request_data)
        print(expected_response)
        self.assertEqual(request_data, expected_response)

        data_converter._api_type = OpenAIAPIType.IMAGE_EDIT
        data_converter._tag_data.task = Task.UNDETECTED
        request_data = data_converter.get_data_package()
        expected_response.task = Task.IMAGE_EDITING
        expected_response.api_type = OpenAIAPIType.IMAGE_EDIT.value
        self.assertEqual(request_data, expected_response)

        data_converter._api_type = OpenAIAPIType.IMAGE_VARIATION
        data_converter._tag_data.task = Task.UNDETECTED
        request_data = data_converter.get_data_package()
        expected_response.task = Task.IMAGE_VARIATION
        expected_response.api_type = OpenAIAPIType.IMAGE_VARIATION.value
        self.assertEqual(request_data, expected_response)

    @patch("nebuly.trackers.openai.openai")
    def test_get_data_package__is_returning_the_correct_data_for_finetune_api(
        self,
        mocked_openai,
    ):
        mocked_openai.api_type = "open_ai"
        mocked_api_type = OpenAIAPIType.FINETUNE
        tag_data = TagData(
            project="unknown_project",
            phase=DevelopmentPhase.UNKNOWN,
            task=Task.UNDETECTED,
        )
        data_converter = OpenaiAIDataPackageConverter(
            request_kwargs=self.finetune_request_kwargs,
            request_response=self.finetune_response,
            api_type=mocked_api_type,
        )
        data_converter._timestamp = self.mocked_timestamp
        data_converter.set_tag_data(tag_data)
        request_data = data_converter.get_data_package()
        expected_response = NebulyDataPackage(
            project="unknown_project",
            phase="unknown",
            task=Task.FINETUNING,
            provider="openai",
            api_type=OpenAIAPIType.FINETUNE.value,
            timestamp=self.mocked_timestamp,
            model="curie",
            n_prompt_tokens=None,
            n_output_tokens=None,
            n_output_images=None,
            image_size=None,
            audio_duration_seconds=None,
            training_file_id="file-XGinujblHPwGLSztz8cPS8XY",
        )
        self.assertEqual(request_data, expected_response)

    @patch("nebuly.trackers.openai.openai")
    def test_get_data_package__is_returning_the_correct_data_for_moderation_api(
        self,
        mocked_openai,
    ):
        mocked_openai.api_type = "open_ai"
        mocked_api_type = OpenAIAPIType.MODERATION
        tag_data = TagData(
            project="unknown_project",
            phase=DevelopmentPhase.UNKNOWN,
            task=Task.UNDETECTED,
        )
        data_converter = OpenaiAIDataPackageConverter(
            request_kwargs=self.moderation_request_kwargs,
            request_response=self.moderation_response,
            api_type=mocked_api_type,
        )
        data_converter._timestamp = self.mocked_timestamp
        data_converter.set_tag_data(tag_data)
        request_data = data_converter.get_data_package()

        expected_response = NebulyDataPackage(
            project="unknown_project",
            phase="unknown",
            task=Task.TEXT_MODERATION,
            provider="openai",
            api_type=OpenAIAPIType.MODERATION.value,
            timestamp=self.mocked_timestamp,
            model="text-moderation-001",
            n_prompt_tokens=None,
            n_output_tokens=None,
            n_output_images=None,
            image_size=None,
            audio_duration_seconds=None,
            training_file_id=None,
        )

        self.assertEqual(request_data, expected_response)


class TestOpenAITracker(unittest.TestCase):
    mocked_openai_response = "OpenAIResponse"
    mocked_kwargs = {"arg": "arg"}

    @patch("nebuly.trackers.openai.openai")
    def test_replace_sdk_functions__is_issuing_the_track_request_for_text_completion(
        self,
        mocked_openai,
    ):
        mocked_openai.api_type = "open_ai"
        mocked_nebuly_queue = MagicMock()
        mocked_openai.Completion.create.return_value = self.mocked_openai_response

        openai_tracker = OpenAITracker(mocked_nebuly_queue)
        openai_tracker.replace_sdk_functions()
        mocked_openai.Completion.create(**self.mocked_kwargs)

        mocked_nebuly_queue.put.assert_called_once()
        args, _ = mocked_nebuly_queue.put.call_args
        queue_object = args[0]
        self.assertEqual(
            queue_object._data_package_converter._request_kwargs, self.mocked_kwargs
        )
        self.assertEqual(
            queue_object._data_package_converter._request_response,
            self.mocked_openai_response,
        )
        self.assertEqual(
            queue_object._data_package_converter._api_type,
            OpenAIAPIType.TEXT_COMPLETION,
        )

    @patch("nebuly.trackers.openai.openai")
    def test_replace_sdk_functions__is_replacing_the_method_for_text_completion(
        self,
        mocked_openai,
    ):
        mocked_nebuly_queue = MagicMock()

        openai_tracker = OpenAITracker(mocked_nebuly_queue)
        openai_tracker.replace_sdk_functions()

        self.assertNotIsInstance(mocked_openai.Completion.create, MagicMock)

    @patch("nebuly.trackers.openai.openai")
    def test_replace_sdk_functions__is_issuing_the_track_request_for_chat_completion(
        self,
        mocked_openai,
    ):
        mocked_openai.api_type = "open_ai"
        mocked_nebuly_queue = MagicMock()
        mocked_openai.ChatCompletion.create.return_value = self.mocked_openai_response

        openai_tracker = OpenAITracker(mocked_nebuly_queue)
        openai_tracker.replace_sdk_functions()
        mocked_openai.ChatCompletion.create(**self.mocked_kwargs)

        mocked_nebuly_queue.put.assert_called_once()
        args, _ = mocked_nebuly_queue.put.call_args
        queue_object = args[0]
        self.assertEqual(
            queue_object._data_package_converter._request_kwargs, self.mocked_kwargs
        )
        self.assertEqual(
            queue_object._data_package_converter._request_response,
            self.mocked_openai_response,
        )
        self.assertEqual(
            queue_object._data_package_converter._api_type, OpenAIAPIType.CHAT
        )

    @patch("nebuly.trackers.openai.openai")
    def test_replace_sdk_functions__is_replacing_the_method_chat_completion(
        self,
        mocked_openai,
    ):
        mocked_nebuly_queue = MagicMock()

        openai_tracker = OpenAITracker(mocked_nebuly_queue)
        openai_tracker.replace_sdk_functions()

        self.assertNotIsInstance(mocked_openai.ChatCompletion.create, MagicMock)

    @patch("nebuly.trackers.openai.openai")
    def test_replace_sdk_functions__is_issuing_the_track_request_for_edit(
        self,
        mocked_openai,
    ):
        mocked_openai.api_type = "open_ai"
        mocked_nebuly_queue = MagicMock()
        mocked_openai.Edit.create.return_value = self.mocked_openai_response

        openai_tracker = OpenAITracker(mocked_nebuly_queue)
        openai_tracker.replace_sdk_functions()
        mocked_openai.Edit.create(**self.mocked_kwargs)

        mocked_nebuly_queue.put.assert_called_once()
        args, _ = mocked_nebuly_queue.put.call_args
        queue_object = args[0]
        self.assertEqual(
            queue_object._data_package_converter._request_kwargs, self.mocked_kwargs
        )
        self.assertEqual(
            queue_object._data_package_converter._request_response,
            self.mocked_openai_response,
        )
        self.assertEqual(
            queue_object._data_package_converter._api_type, OpenAIAPIType.EDIT
        )

    @patch("nebuly.trackers.openai.openai")
    def test_replace_sdk_functions__is_replacing_the_method_edit(
        self,
        mocked_openai,
    ):
        mocked_nebuly_queue = MagicMock()

        openai_tracker = OpenAITracker(mocked_nebuly_queue)
        openai_tracker.replace_sdk_functions()

        self.assertNotIsInstance(mocked_openai.Edit.create, MagicMock)

    @patch("nebuly.trackers.openai.openai")
    def test_replace_sdk_functions__is_issuing_the_track_request_for_embedding(
        self,
        mocked_openai,
    ):
        mocked_openai.api_type = "open_ai"
        mocked_nebuly_queue = MagicMock()
        mocked_openai.Embedding.create.return_value = self.mocked_openai_response

        openai_tracker = OpenAITracker(mocked_nebuly_queue)
        openai_tracker.replace_sdk_functions()
        mocked_openai.Embedding.create(**self.mocked_kwargs)

        mocked_nebuly_queue.put.assert_called_once()
        args, _ = mocked_nebuly_queue.put.call_args
        queue_object = args[0]
        self.assertEqual(
            queue_object._data_package_converter._request_kwargs, self.mocked_kwargs
        )
        self.assertEqual(
            queue_object._data_package_converter._request_response,
            self.mocked_openai_response,
        )
        self.assertEqual(
            queue_object._data_package_converter._api_type, OpenAIAPIType.EMBEDDING
        )

    @patch("nebuly.trackers.openai.openai")
    def test_replace_sdk_functions__is_replacing_the_method_embedding(
        self,
        mocked_openai,
    ):
        mocked_nebuly_queue = MagicMock()

        openai_tracker = OpenAITracker(mocked_nebuly_queue)
        openai_tracker.replace_sdk_functions()

        self.assertNotIsInstance(mocked_openai.Embedding.create, MagicMock)

    @patch("nebuly.trackers.openai.openai")
    def test_replace_sdk_functions__is_issuing_the_track_request_for_audio_transcribe(
        self,
        mocked_openai,
    ):
        mocked_openai.api_type = "open_ai"
        mocked_nebuly_queue = MagicMock()
        mocked_openai.Audio.transcribe.return_value = self.mocked_openai_response

        openai_tracker = OpenAITracker(mocked_nebuly_queue)
        openai_tracker.replace_sdk_functions()
        mocked_openai.Audio.transcribe(**self.mocked_kwargs)

        mocked_nebuly_queue.put.assert_called_once()
        args, _ = mocked_nebuly_queue.put.call_args
        queue_object = args[0]
        self.assertEqual(
            queue_object._data_package_converter._request_kwargs, self.mocked_kwargs
        )
        self.assertEqual(
            queue_object._data_package_converter._request_response,
            self.mocked_openai_response,
        )
        self.assertEqual(
            queue_object._data_package_converter._api_type,
            OpenAIAPIType.AUDIO_TRANSCRIBE,
        )

    @patch("nebuly.trackers.openai.openai")
    def test_replace_sdk_functions__is_replacing_the_method_audio_transcribe(
        self,
        mocked_openai,
    ):
        mocked_nebuly_queue = MagicMock()

        openai_tracker = OpenAITracker(mocked_nebuly_queue)
        openai_tracker.replace_sdk_functions()

        self.assertNotIsInstance(mocked_openai.Audio.transcribe, MagicMock)

    @patch("nebuly.trackers.openai.openai")
    def test_replace_sdk_functions__is_issuing_the_track_request_for_audio_translate(
        self,
        mocked_openai,
    ):
        mocked_openai.api_type = "open_ai"
        mocked_nebuly_queue = MagicMock()
        mocked_openai.Audio.translate.return_value = self.mocked_openai_response

        openai_tracker = OpenAITracker(mocked_nebuly_queue)
        openai_tracker.replace_sdk_functions()
        mocked_openai.Audio.translate(**self.mocked_kwargs)

        mocked_nebuly_queue.put.assert_called_once()
        args, _ = mocked_nebuly_queue.put.call_args
        queue_object = args[0]
        self.assertEqual(
            queue_object._data_package_converter._request_kwargs, self.mocked_kwargs
        )
        self.assertEqual(
            queue_object._data_package_converter._request_response,
            self.mocked_openai_response,
        )
        self.assertEqual(
            queue_object._data_package_converter._api_type,
            OpenAIAPIType.AUDIO_TRANSLATE,
        )

    @patch("nebuly.trackers.openai.openai")
    def test_replace_sdk_functions__is_replacing_the_method_for_audio_translate(
        self,
        mocked_openai,
    ):
        mocked_nebuly_queue = MagicMock()

        openai_tracker = OpenAITracker(mocked_nebuly_queue)
        openai_tracker.replace_sdk_functions()

        self.assertNotIsInstance(mocked_openai.Audio.translate, MagicMock)

    @patch("nebuly.trackers.openai.openai")
    def test_replace_sdk_functions__is_issuing_the_track_request_for_image_create(
        self,
        mocked_openai,
    ):
        mocked_openai.api_type = "open_ai"
        mocked_nebuly_queue = MagicMock()
        mocked_openai.Image.create.return_value = self.mocked_openai_response

        openai_tracker = OpenAITracker(mocked_nebuly_queue)
        openai_tracker.replace_sdk_functions()
        mocked_openai.Image.create(**self.mocked_kwargs)

        mocked_nebuly_queue.put.assert_called_once()
        args, _ = mocked_nebuly_queue.put.call_args
        queue_object = args[0]
        self.assertEqual(
            queue_object._data_package_converter._request_kwargs, self.mocked_kwargs
        )
        self.assertEqual(
            queue_object._data_package_converter._request_response,
            self.mocked_openai_response,
        )
        self.assertEqual(
            queue_object._data_package_converter._api_type, OpenAIAPIType.IMAGE_CREATE
        )

    @patch("nebuly.trackers.openai.openai")
    def test_replace_sdk_functions__is_replacing_the_method_image_create(
        self,
        mocked_openai,
    ):
        mocked_nebuly_queue = MagicMock()

        openai_tracker = OpenAITracker(mocked_nebuly_queue)
        openai_tracker.replace_sdk_functions()

        self.assertNotIsInstance(mocked_openai.Image.create, MagicMock)

    @patch("nebuly.trackers.openai.openai")
    def test_replace_sdk_functions__is_issuing_the_track_request_for_image_edit(
        self,
        mocked_openai,
    ):
        mocked_openai.api_type = "open_ai"
        mocked_nebuly_queue = MagicMock()
        mocked_openai.Image.create_edit.return_value = self.mocked_openai_response

        openai_tracker = OpenAITracker(mocked_nebuly_queue)
        openai_tracker.replace_sdk_functions()
        mocked_openai.Image.create_edit(**self.mocked_kwargs)

        mocked_nebuly_queue.put.assert_called_once()
        args, _ = mocked_nebuly_queue.put.call_args
        queue_object = args[0]
        self.assertEqual(
            queue_object._data_package_converter._request_kwargs, self.mocked_kwargs
        )
        self.assertEqual(
            queue_object._data_package_converter._request_response,
            self.mocked_openai_response,
        )
        self.assertEqual(
            queue_object._data_package_converter._api_type, OpenAIAPIType.IMAGE_EDIT
        )

    @patch("nebuly.trackers.openai.openai")
    def test_replace_sdk_functions__is_replacing_the_method_image_edit(
        self,
        mocked_openai,
    ):
        mocked_nebuly_queue = MagicMock()

        openai_tracker = OpenAITracker(mocked_nebuly_queue)
        openai_tracker.replace_sdk_functions()

        self.assertNotIsInstance(mocked_openai.Image.create_edit, MagicMock)

    @patch("nebuly.trackers.openai.openai")
    def test_replace_sdk_functions__is_issuing_the_track_request_for_image_variation(
        self,
        mocked_openai,
    ):
        mocked_openai.api_type = "open_ai"
        mocked_nebuly_queue = MagicMock()
        mocked_openai.Image.create_variation.return_value = self.mocked_openai_response

        openai_tracker = OpenAITracker(mocked_nebuly_queue)
        openai_tracker.replace_sdk_functions()
        mocked_openai.Image.create_variation(**self.mocked_kwargs)

        mocked_nebuly_queue.put.assert_called_once()
        args, _ = mocked_nebuly_queue.put.call_args
        queue_object = args[0]
        self.assertEqual(
            queue_object._data_package_converter._request_kwargs, self.mocked_kwargs
        )
        self.assertEqual(
            queue_object._data_package_converter._request_response,
            self.mocked_openai_response,
        )
        self.assertEqual(
            queue_object._data_package_converter._api_type,
            OpenAIAPIType.IMAGE_VARIATION,
        )

    @patch("nebuly.trackers.openai.openai")
    def test_replace_sdk_functions__is_replacing_the_method_image_variation(
        self,
        mocked_openai,
    ):
        mocked_nebuly_queue = MagicMock()

        openai_tracker = OpenAITracker(mocked_nebuly_queue)
        openai_tracker.replace_sdk_functions()

        self.assertNotIsInstance(mocked_openai.Image.create_variation, MagicMock)

    @patch("nebuly.trackers.openai.openai")
    def test_replace_sdk_functions__is_issuing_the_track_request_for_finetune(
        self,
        mocked_openai,
    ):
        mocked_openai.api_type = "open_ai"
        mocked_nebuly_queue = MagicMock()
        mocked_openai.FineTune.create.return_value = self.mocked_openai_response

        openai_tracker = OpenAITracker(mocked_nebuly_queue)
        openai_tracker.replace_sdk_functions()
        mocked_openai.FineTune.create(**self.mocked_kwargs)

        mocked_nebuly_queue.put.assert_called_once()
        args, _ = mocked_nebuly_queue.put.call_args
        queue_object = args[0]
        self.assertEqual(
            queue_object._data_package_converter._request_kwargs, self.mocked_kwargs
        )
        self.assertEqual(
            queue_object._data_package_converter._request_response,
            self.mocked_openai_response,
        )
        self.assertEqual(
            queue_object._data_package_converter._api_type, OpenAIAPIType.FINETUNE
        )

    @patch("nebuly.trackers.openai.openai")
    def test_replace_sdk_functions__is_replacing_the_method_finetune(
        self,
        mocked_openai,
    ):
        mocked_nebuly_queue = MagicMock()

        openai_tracker = OpenAITracker(mocked_nebuly_queue)
        openai_tracker.replace_sdk_functions()

        self.assertNotIsInstance(mocked_openai.FineTune.create, MagicMock)

    @patch("nebuly.trackers.openai.openai")
    def test_replace_sdk_functions__is_issuing_the_track_request_for_moderation(
        self,
        mocked_openai,
    ):
        mocked_openai.api_type = "open_ai"
        mocked_nebuly_queue = MagicMock()
        mocked_openai.Moderation.create.return_value = self.mocked_openai_response

        openai_tracker = OpenAITracker(mocked_nebuly_queue)
        openai_tracker.replace_sdk_functions()
        mocked_openai.Moderation.create(**self.mocked_kwargs)

        mocked_nebuly_queue.put.assert_called_once()
        args, _ = mocked_nebuly_queue.put.call_args
        queue_object = args[0]
        self.assertEqual(
            queue_object._data_package_converter._request_kwargs, self.mocked_kwargs
        )
        self.assertEqual(
            queue_object._data_package_converter._request_response,
            self.mocked_openai_response,
        )
        self.assertEqual(
            queue_object._data_package_converter._api_type, OpenAIAPIType.MODERATION
        )

    @patch("nebuly.trackers.openai.openai")
    def test_replace_sdk_functions__is_replacing_the_method_moderation(
        self,
        mocked_openai,
    ):
        mocked_nebuly_queue = MagicMock()

        openai_tracker = OpenAITracker(mocked_nebuly_queue)
        openai_tracker.replace_sdk_functions()

        self.assertNotIsInstance(mocked_openai.Moderation.create, MagicMock)
