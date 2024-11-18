from __future__ import annotations

import os
import ssl

from nebuly.entities import Package, SupportedVersion

NEBULY_KWARGS = {
    "user_id",
    "user_group_profile",
    "parent_run_id",
    "root_run_id",
    "nebuly_tags",
    "nebuly_interaction",
    "feature_flags",
    "nebuly_api_key",
}

PACKAGES = (
    Package(
        "openai",
        SupportedVersion("0.10.0", "0.30.0"),
        (
            "Completion.create",
            "Completion.acreate",
            "ChatCompletion.create",
            "ChatCompletion.acreate",
        ),
    ),
    Package(
        "openai",
        SupportedVersion("1.0.0", "1.21.0"),
        (
            "resources.chat.completions.Completions.create",
            "resources.chat.completions.AsyncCompletions.create",
            "resources.completions.Completions.create",
            "resources.completions.AsyncCompletions.create",
            "resources.beta.threads.messages.messages.Messages.list",
            "resources.beta.threads.messages.messages.AsyncMessages.list",
        ),
    ),
    Package(
        "openai",
        SupportedVersion("1.21.0"),
        (
            "resources.chat.completions.Completions.create",
            "resources.chat.completions.AsyncCompletions.create",
            "resources.completions.Completions.create",
            "resources.completions.AsyncCompletions.create",
            "resources.beta.threads.messages.Messages.list",
            "resources.beta.threads.messages.AsyncMessages.list",
        ),
    ),
    Package(
        "cohere",
        SupportedVersion("4.0.0"),
        (
            "Client.generate",
            "AsyncClient.generate",
            "Client.chat",
            "AsyncClient.chat",
        ),
    ),
    Package(
        "anthropic",
        SupportedVersion("0.3.0"),
        (
            "resources.Completions.create",
            "resources.AsyncCompletions.create",
        ),
    ),
    Package(
        "huggingface_hub",
        SupportedVersion("0.12.0"),
        (
            "InferenceClient.conversational",
            "InferenceClient.text_generation",
            "AsyncInferenceClient.conversational",
            "AsyncInferenceClient.text_generation",
        ),
    ),
    Package(
        "transformers",
        SupportedVersion("4.10.0"),
        (
            "pipelines.conversational.ConversationalPipeline.__call__",
            "pipelines.text_generation.TextGenerationPipeline.__call__",
        ),
    ),
    Package(
        "google",
        SupportedVersion("0.0.1"),
        (
            "generativeai.generate_text",
            "generativeai.chat",
            "generativeai.chat_async",
            "generativeai.discuss.ChatResponse.reply",
        ),
    ),
    Package(
        "vertexai",
        SupportedVersion("0.0.1"),
        (
            "language_models.TextGenerationModel.predict",
            "language_models.TextGenerationModel.predict_async",
            "language_models.TextGenerationModel.predict_streaming",
            "language_models.ChatSession.send_message",
            "language_models.ChatSession.send_message_async",
            "language_models.ChatSession.send_message_streaming",
        ),
    ),
    Package(
        "botocore",
        SupportedVersion("1.30.0"),
        ("client.BaseClient._make_api_call",),
    ),
)


def get_ssl_verify_mode() -> ssl.VerifyMode:
    verify_mode = os.getenv("NEBULY_SSL_VERIFY_MODE", "CERT_REQUIRED")
    if verify_mode == "CERT_OPTIONAL":
        return ssl.CERT_OPTIONAL
    if verify_mode == "CERT_REQUIRED":
        return ssl.CERT_REQUIRED
    if verify_mode == "CERT_NONE":
        return ssl.CERT_NONE
    raise ValueError(
        f"Invalid value for NEBULY_SSL_VERIFY_MODE: {verify_mode}. "
        f"Supported values are: CERT_OPTIONAL, CERT_REQUIRED, CERT_NONE."
    )