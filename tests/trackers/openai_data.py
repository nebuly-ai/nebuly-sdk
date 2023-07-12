from dataclasses import dataclass


@dataclass
class TestData:
    request_kwargs: dict
    request_response: dict


text_completion = TestData(
    request_kwargs={
        "model": "text-davinci-003",
        "prompt": "Say this is a test",
        "max_tokens": 7,
        "temperature": 0,
        "top_p": 1,
        "n": 1,
        "stream": False,
        "logprobs": None,
        "stop": "\n",
        "user": "user-123",
    },
    request_response={
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
    },
)

text_completion_stream = TestData(
    request_kwargs={
        "model": "text-davinci-003",
        "prompt": "Say this is a test",
        "max_tokens": 7,
        "temperature": 0,
        "top_p": 1,
        "n": 1,
        "stream": True,
        "logprobs": None,
        "stop": "\n",
        "user": "user-123",
    },
    request_response={
        "output_text": "\n\nThis is indeed a test",
    },
)

text_completion_generator_response = {
    "choices": [{"finish_reason": None, "index": 0, "logprobs": None, "text": " a"}],
    "created": 1686461688,
    "id": "cmpl-7Q80uDV1zpqrUhEd5LDUou0jiLmMz",
    "model": "text-davinci-003",
    "object": "text_completion",
    "user": "user-123",
}

chat_completion = TestData(
    request_kwargs={
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello!"}],
        "user": "user-123",
    },
    request_response={
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "\n\nHello there, how may I assist you today?",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
    },
)

chat_completion_stream = TestData(
    request_kwargs={
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello!"}],
        "stream": True,
        "user": "user-123",
    },
    request_response={
        "output_text": "\n\nHello there, how may I assist you today?",
    },
)

chat_completion_generator_response = {
    "choices": [{"delta": {"content": "\n\n"}, "finish_reason": None, "index": 0}],
    "created": 1677825464,
    "id": "chatcmpl-6ptKyqKOGXZT6iQnqiXAH8adNLUzD",
    "model": "gpt-3.5-turbo-0301",
    "object": "chat.completion.chunk",
}

edit = TestData(
    request_kwargs={
        "model": "text-davinci-edit-001",
        "input": "What day of the wek is it?",
        "instruction": "Fix the spelling mistakes",
        "user": "user-123",
    },
    request_response={
        "object": "edit",
        "created": 1589478378,
        "choices": [
            {
                "text": "What day of the week is it?",
                "index": 0,
            }
        ],
        "usage": {"prompt_tokens": 25, "completion_tokens": 32, "total_tokens": 57},
    },
)

image = TestData(
    request_kwargs={
        "prompt": "A cute baby sea otter",
        "n": 2,
        "size": "1024x1024",
        "user": "user-123",
    },
    request_response={
        "created": 1589478378,
        "data": [{"url": "https://..."}, {"url": "https://..."}],
    },
)

embedding = TestData(
    request_kwargs={
        "model": "text-embedding-ada-002",
        "input": "The food was delicious and the waiter...",
        "user": "user-123",
    },
    request_response={
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": [
                    0.0023064255,
                    -0.009327292,
                    -0.0028842222,
                ],
                "index": 0,
            }
        ],
        "model": "text-embedding-ada-002",
        "usage": {"prompt_tokens": 8, "total_tokens": 8},
    },
)

audio = TestData(
    request_kwargs={
        "file": "audio.mp3",
        "model": "whisper-1",
        "user": "user-123",
    },
    request_response={
        "text": """
        Imagine the wildest idea that you've ever had,
        and you're curious about how it might scale to
        something that's a 100, a 1,000 times bigger.
        This is a place where you can get to do that.
        """
    },
)

finetune = TestData(
    request_kwargs={
        "training_file": "file-XGinujblHPwGLSztz8cPS8XY",
        "user": "user-123",
    },
    request_response={
        "id": "ft-AF1WoRqd3aJAHsqc9NY7iL8F",
        "object": "fine-tune",
        "model": "curie",
        "created_at": 1614807352,
        "events": [
            {
                "object": "fine-tune-event",
                "created_at": 1614807352,
                "level": "info",
                "message": """
                Job enqueued. Waiting for jobs ahead to complete.
                Queue number: 0.""",
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
    },
)

moderation = TestData(
    request_kwargs={
        "input": "Some test input.",
        "user": "user-123",
    },
    request_response={
        "id": "modr-5MWoLO",
        "model": "text-moderation-001",
        "results": [
            {
                "categories": {
                    "hate": False,
                    "hate/threatening": True,
                    "self-harm": False,
                    "sexual": False,
                    "sexual/minors": False,
                    "violence": True,
                    "violence/graphic": False,
                },
                "category_scores": {
                    "hate": 0.22714105248451233,
                    "hate/threatening": 0.4132447838783264,
                    "self-harm": 0.005232391878962517,
                    "sexual": 0.01407341007143259,
                    "sexual/minors": 0.0038522258400917053,
                    "violence": 0.9223177433013916,
                    "violence/graphic": 0.036865197122097015,
                },
                "flagged": True,
            }
        ],
    },
)
