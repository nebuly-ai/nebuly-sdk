from __future__ import annotations

from nebuly.entities import Package

PACKAGES = (
    Package(
        "openai",
        ("0.10.2",),
        (
            "Completion.create",
            "Completion.acreate",
            "ChatCompletion.create",
            "ChatCompletion.acreate",
        ),
    ),
    Package(
        "langchain",
        ("0.0.200",),
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
        ("4.0.0",),
        (
            "Client.generate",
            "AsyncClient.generate",
            "Client.chat",
            "AsyncClient.chat",
        ),
    ),
    Package(
        "anthropic",
        ("0.3.0",),
        (
            "resources.Completions.create",
            "resources.AsyncCompletions.create",
        ),
    ),
    Package("huggingface_hub", ("0.12.0",), ("InferenceClient.conversational",)),
    Package(
        "google",
        ("0.0.1",),
        ("generativeai.generate_text",),
    ),
)
