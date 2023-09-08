from __future__ import annotations

from nebuly.entities import Package

PACKAGES = (
    Package(
        "openai",
        ("0.10.2",),
        (
            "Completion.create",
            "Completion.create",
            "ChatCompletion.create",
            "Edit.create",
            "Embedding.create",
            "FineTune.create",
            "Moderation.create",
        ),
    ),
)
