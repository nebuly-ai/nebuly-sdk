from __future__ import annotations

from nebuly.entities import Package, SupportedVersion

NEBULY_KWARGS = {
    "user_id",
    "user_group_profile",
    "parent_run_id",
    "root_run_id",
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
        SupportedVersion("1.0.0"),
        (
            "resources.chat.completions.Completions.create",
            "resources.chat.completions.AsyncCompletions.create",
            "resources.completions.Completions.create",
            "resources.completions.AsyncCompletions.create",
        ),
    ),
    Package(
        "langchain",
        SupportedVersion("0.0.200"),
        (
            "chains.base.Chain.__call__",
            "chains.base.Chain.acall",
            "llms.base.BaseLLM.generate",
            "llms.base.BaseLLM.agenerate",
            "chat_models.base.BaseChatModel.generate",
            "chat_models.base.BaseChatModel.agenerate",
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
        ("pipelines.base.Pipeline.__call__",),
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
)
