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
            "Embedding.create",
            "Embedding.acreate",
            "Moderation.create",
            "Moderation.acreate",
            "FineTuningJob.create",
            "FineTuningJob.acreate",
        ),
    ),
)
