# pylint: disable=duplicate-code
from __future__ import annotations

import json
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import boto3  # type: ignore
import pytest
import urllib3
from botocore.eventstream import EventStream  # type: ignore
from botocore.response import StreamingBody  # type: ignore

from nebuly.entities import InteractionWatch, SpanWatch
from nebuly.observers import NebulyObserver
from nebuly.requests import CustomJSONEncoder
from tests.providers.common import nebuly_init


@pytest.fixture(name="a2i_completion")
def fixture_aws_bedrock_a2i_invoke_model() -> dict[str, Any]:
    raw_stream = MagicMock(
        spec=urllib3.response.HTTPResponse,
        data=b'{"id":1234,"prompt":{"text":"Say \'HI\'","tokens":[{"generatedToken":{"token":"\xe2\x96\x81Say","logprob":-10.174832344055176,"raw_logprob":-10.174832344055176},"topTokens":null,"textRange":{"start":0,"end":3}},{"generatedToken":{"token":"\xe2\x96\x81\'","logprob":-5.495139122009277,"raw_logprob":-5.495139122009277},"topTokens":null,"textRange":{"start":3,"end":5}},{"generatedToken":{"token":"HI","logprob":-7.433229446411133,"raw_logprob":-7.433229446411133},"topTokens":null,"textRange":{"start":5,"end":7}},{"generatedToken":{"token":"\'","logprob":-0.07328788191080093,"raw_logprob":-0.07328788191080093},"topTokens":null,"textRange":{"start":7,"end":8}}]},"completions":[{"data":{"text":"\\nHi!","tokens":[{"generatedToken":{"token":"<|newline|>","logprob":0.0,"raw_logprob":-0.09085399657487869},"topTokens":null,"textRange":{"start":0,"end":1}},{"generatedToken":{"token":"\xe2\x96\x81Hi","logprob":0.0,"raw_logprob":-1.158215045928955},"topTokens":null,"textRange":{"start":1,"end":3}},{"generatedToken":{"token":"!","logprob":0.0,"raw_logprob":-0.05865238606929779},"topTokens":null,"textRange":{"start":3,"end":4}},{"generatedToken":{"token":"<|endoftext|>","logprob":0.0,"raw_logprob":-0.4792883098125458},"topTokens":null,"textRange":{"start":4,"end":4}}]},"finishReason":{"reason":"endoftext"}}]}',  # pylint: disable=line-too-long  # noqa: E501
    )
    return {
        "ResponseMetadata": {
            "RequestId": "830dfcc1-33f5-4b3e-bf84-7c519715ff30",
            "HTTPStatusCode": 200,
            "HTTPHeaders": {
                "date": "Mon, 16 Oct 2023 14:58:21 GMT",
                "content-type": "application/json",
                "content-length": "10132",
                "connection": "keep-alive",
                "x-amzn-requestid": "830dfcc1-33f5-4b3e-bf84-7c519715ff30",
            },
            "RetryAttempts": 0,
        },
        "contentType": "application/json",
        "body": StreamingBody(
            raw_stream=raw_stream,
            content_length=1280,
        ),
    }


def test_aws_bedrock_invoke_model__a2i(
    a2i_completion: dict[str, Any],
) -> None:
    with patch("botocore.client.BaseClient._make_api_call") as mock_completion:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion.return_value = a2i_completion
            nebuly_init(observer=mock_observer)

            bedrock = boto3.client(
                service_name="bedrock-runtime", region_name="us-east-1"
            )
            body = json.dumps(
                {
                    "prompt": "Say 'HI'",
                    "maxTokens": 200,
                    "temperature": 0.5,
                    "topP": 0.5,
                }
            )

            modelId = "ai21.j2-mid-v1"
            accept = "application/json"
            contentType = "application/json"

            response = bedrock.invoke_model(
                body=body,
                modelId=modelId,
                accept=accept,
                contentType=contentType,
                user_id="test_user",
                user_group_profile="test_user_group_profile",
            )

            assert response is not None
            assert (
                response.get("body").read()
                == a2i_completion[  # pylint: disable=protected-access
                    "body"
                ]._raw_stream.data
            )
            assert mock_observer.call_count == 1

            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.end_user == "test_user"
            assert interaction_watch.end_user_group_profile == "test_user_group_profile"
            assert interaction_watch.input == "Say 'HI'"
            assert interaction_watch.output == "\nHi!"
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


@pytest.fixture(name="anthropic_completion")
def fixture_aws_bedrock_invoke_model_anthropic() -> dict[str, Any]:
    raw_stream = MagicMock(
        spec=urllib3.response.HTTPResponse,
        data=b'{"completion":" Hello!","stop_reason":"stop_sequence"}',
    )
    return {
        "ResponseMetadata": {
            "RequestId": "830dfcc1-33f5-4b3e-bf84-7c519715ff30",
            "HTTPStatusCode": 200,
            "HTTPHeaders": {
                "date": "Mon, 16 Oct 2023 14:58:21 GMT",
                "content-type": "application/json",
                "content-length": "10132",
                "connection": "keep-alive",
                "x-amzn-requestid": "830dfcc1-33f5-4b3e-bf84-7c519715ff30",
            },
            "RetryAttempts": 0,
        },
        "contentType": "application/json",
        "body": StreamingBody(
            raw_stream=raw_stream,
            content_length=1280,
        ),
    }


def test_aws_bedrock_invoke_model__anthropic(
    anthropic_completion: dict[str, Any],
) -> None:
    with patch("botocore.client.BaseClient._make_api_call") as mock_completion:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion.return_value = anthropic_completion
            nebuly_init(observer=mock_observer)

            bedrock = boto3.client(
                service_name="bedrock-runtime", region_name="us-east-1"
            )
            body = json.dumps(
                {
                    "prompt": "\n\nHuman:say hi\n\nAssistant:",
                    "max_tokens_to_sample": 300,
                }
            )

            modelId = "anthropic.claude-v2"
            accept = "application/json"
            contentType = "application/json"

            response = bedrock.invoke_model(
                body=body,
                modelId=modelId,
                accept=accept,
                contentType=contentType,
                user_id="test_user",
                user_group_profile="test_user_group_profile",
            )

            assert response is not None
            assert (
                response.get("body").read()
                == anthropic_completion[  # pylint: disable=protected-access
                    "body"
                ]._raw_stream.data
            )
            assert mock_observer.call_count == 1

            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.end_user == "test_user"
            assert interaction_watch.end_user_group_profile == "test_user_group_profile"
            assert interaction_watch.input == "say hi"
            assert interaction_watch.output == " Hello!"
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


@pytest.fixture(name="cohere_completion")
def fixture_aws_bedrock_cohere_invoke_model() -> dict[str, Any]:
    raw_stream = MagicMock(
        spec=urllib3.response.HTTPResponse,
        data=b'{"generations":[{"id":"101d97be-392c-44a6-8493-fad243154941","text":"'
        b'Hi! How can I help you today?"}],"id":"c117e48d-e39f-4e9f-b965-4ab8520'
        b'24f91","prompt":"\\n\\nHuman:say hi\\n\\nAssistant:"}',
    )
    return {
        "ResponseMetadata": {
            "RequestId": "830dfcc1-33f5-4b3e-bf84-7c519715ff30",
            "HTTPStatusCode": 200,
            "HTTPHeaders": {
                "date": "Mon, 16 Oct 2023 14:58:21 GMT",
                "content-type": "application/json",
                "content-length": "10132",
                "connection": "keep-alive",
                "x-amzn-requestid": "830dfcc1-33f5-4b3e-bf84-7c519715ff30",
            },
            "RetryAttempts": 0,
        },
        "contentType": "application/json",
        "body": StreamingBody(
            raw_stream=raw_stream,
            content_length=189,
        ),
    }


def test_aws_bedrock_invoke_model__cohere(
    cohere_completion: dict[str, Any],
) -> None:
    with patch("botocore.client.BaseClient._make_api_call") as mock_completion:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion.return_value = cohere_completion
            nebuly_init(observer=mock_observer)

            bedrock = boto3.client(
                service_name="bedrock-runtime", region_name="us-east-1"
            )
            body = json.dumps(
                {
                    "prompt": "Say 'HI'",
                }
            )

            modelId = "cohere.command-text-v14"
            accept = "application/json"
            contentType = "application/json"

            response = bedrock.invoke_model(
                body=body,
                modelId=modelId,
                accept=accept,
                contentType=contentType,
                user_id="test_user",
                user_group_profile="test_user_group_profile",
            )

            assert response is not None
            assert (
                response.get("body").read()
                == cohere_completion[  # pylint: disable=protected-access
                    "body"
                ]._raw_stream.data
            )
            assert mock_observer.call_count == 1

            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.end_user == "test_user"
            assert interaction_watch.end_user_group_profile == "test_user_group_profile"
            assert interaction_watch.input == "Say 'HI'"
            assert interaction_watch.output == "Hi! How can I help you today?"
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


@pytest.fixture(name="anthropic_completion_stream")
def fixture_aws_bedrock_invoke_model_stream_anthropic() -> (
    Generator[dict[str, Any], None, None]
):  # pylint: disable=line-too-long  # noqa: E501
    event_stream = EventStream(
        raw_stream=MagicMock(spec=urllib3.response.HTTPResponse),
        output_shape=None,
        parser=None,
        operation_name="InvokeModelWithResponseStream",
    )

    with patch.object(EventStream, "__iter__") as mock_iter:
        mock_iter.return_value = iter(
            (
                {"chunk": {"bytes": b'{"completion":" Hello","stop_reason":null}'}},
                {
                    "chunk": {
                        "bytes": b'{"completion":"!","stop_reason":"stop_sequence"}'
                    }
                },
                {
                    "chunk": {
                        "bytes": b'{"completion":"!","stop_reason":"stop_sequence"}'
                    }
                },
            )
        )
        yield {
            "ResponseMetadata": {
                "RequestId": "830dfcc1-33f5-4b3e-bf84-7c519715ff30",
                "HTTPStatusCode": 200,
                "HTTPHeaders": {
                    "date": "Mon, 16 Oct 2023 14:58:21 GMT",
                    "content-type": "application/json",
                    "content-length": "10132",
                    "connection": "keep-alive",
                    "x-amzn-requestid": "830dfcc1-33f5-4b3e-bf84-7c519715ff30",
                },
                "RetryAttempts": 0,
            },
            "contentType": "application/json",
            "body": event_stream,
        }


def test_aws_bedrock_invoke_model_stream__anthropic(
    anthropic_completion_stream: dict[str, Any]
) -> None:
    with patch("botocore.client.BaseClient._make_api_call") as mock_completion:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion.return_value = anthropic_completion_stream
            nebuly_init(observer=mock_observer)

            bedrock = boto3.client(
                service_name="bedrock-runtime", region_name="us-east-1"
            )

            body = json.dumps(
                {
                    "prompt": "\n\nHuman:Say hi\n\nAssistant:",
                    "max_tokens_to_sample": 100,
                }
            )

            result = bedrock.invoke_model_with_response_stream(
                modelId="anthropic.claude-v2",
                body=body,
                user_id="test_user",
                user_group_profile="test_user_group_profile",
            )

            assert list(result.get("body")) is not None
            assert mock_observer.call_count == 1

            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.end_user == "test_user"
            assert (
                interaction_watch.end_user_group_profile == "test_user_group_profile"
            )  # pylint: disable=line-too-long  # noqa: E501
            assert interaction_watch.input == "Say hi"
            assert interaction_watch.output == " Hello!!"
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )


@pytest.fixture(name="cohere_completion_stream")
def fixture_aws_bedrock_invoke_model_stream_cohere() -> (
    Generator[dict[str, Any], None, None]
):  # pylint: disable=line-too-long  # noqa: E501
    event_stream = EventStream(
        raw_stream=MagicMock(spec=urllib3.response.HTTPResponse),
        output_shape=None,
        parser=None,
        operation_name="InvokeModelWithResponseStream",
    )

    with patch.object(EventStream, "__iter__") as mock_iter:
        mock_iter.return_value = iter(
            (
                {"chunk": {"bytes": b'{"is_finished":false,"text":" Hi"}'}},
                {"chunk": {"bytes": b'{"is_finished":false,"text":" there"}'}},
                {"chunk": {"bytes": b'{"is_finished":false,"text":"!"}'}},
                {"chunk": {"bytes": b'{"is_finished":false,"text":" How"}'}},
                {"chunk": {"bytes": b'{"is_finished":false,"text":" can"}'}},
                {"chunk": {"bytes": b'{"is_finished":false,"text":" I"}'}},
                {"chunk": {"bytes": b'{"is_finished":false,"text":" assist"}'}},
                {"chunk": {"bytes": b'{"is_finished":false,"text":" you"}'}},
                {"chunk": {"bytes": b'{"is_finished":false,"text":" today"}'}},
                {"chunk": {"bytes": b'{"is_finished":false,"text":"?"}'}},
                {"chunk": {"bytes": b'{"is_finished":false,"text":"<EOS_TOKEN>"}'}},
                {
                    "chunk": {
                        "bytes": b'{"finish_reason":"COMPLETE","is_finished":true}'
                    }
                },
            )
        )
        yield {
            "ResponseMetadata": {
                "RequestId": "830dfcc1-33f5-4b3e-bf84-7c519715ff30",
                "HTTPStatusCode": 200,
                "HTTPHeaders": {
                    "date": "Mon, 16 Oct 2023 14:58:21 GMT",
                    "content-type": "application/json",
                    "content-length": "10132",
                    "connection": "keep-alive",
                    "x-amzn-requestid": "830dfcc1-33f5-4b3e-bf84-7c519715ff30",
                },
                "RetryAttempts": 0,
            },
            "contentType": "application/json",
            "body": event_stream,
        }


def test_aws_bedrock_invoke_model_stream__cohere(
    cohere_completion_stream: dict[str, Any]
) -> None:
    with patch("botocore.client.BaseClient._make_api_call") as mock_completion:
        with patch.object(NebulyObserver, "on_event_received") as mock_observer:
            mock_completion.return_value = cohere_completion_stream
            nebuly_init(observer=mock_observer)

            bedrock = boto3.client(
                service_name="bedrock-runtime", region_name="us-east-1"
            )

            body = json.dumps(
                {
                    "prompt": "Say hi",
                    "stream": True,
                }
            )

            result = bedrock.invoke_model_with_response_stream(
                modelId="cohere.command-text-v14",
                body=body,
                user_id="test_user",
                user_group_profile="test_user_group_profile",
            )

            assert list(result.get("body")) is not None
            assert mock_observer.call_count == 1

            interaction_watch = mock_observer.call_args[0][0]
            assert isinstance(interaction_watch, InteractionWatch)
            assert interaction_watch.end_user == "test_user"
            assert (
                interaction_watch.end_user_group_profile == "test_user_group_profile"
            )  # pylint: disable=line-too-long  # noqa: E501
            assert interaction_watch.input == "Say hi"
            assert interaction_watch.output == " Hi there! How can I assist you today?"
            assert len(interaction_watch.spans) == 1
            span = interaction_watch.spans[0]
            assert isinstance(span, SpanWatch)
            assert (
                json.dumps(interaction_watch.to_dict(), cls=CustomJSONEncoder)
                is not None
            )
