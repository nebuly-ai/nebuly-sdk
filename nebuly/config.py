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
            "Edit.create",
            "Edit.acreate",
            "Embedding.create",
            "Embedding.acreate",
            "FineTune.create",
            "FineTune.acreate",
            "Moderation.create",
            "Moderation.acreate",
            "FineTunningJob.create",
            "FineTunningJob.acreate",
        ),
    ),
)
