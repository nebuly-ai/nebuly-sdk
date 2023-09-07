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
            "Image.create",
            "Image.create_edit",
            "Image.create_variation",
            "Embedding.create",
            "Audio.transcribe",
            "Audio.translate",
            "FineTune.create",
            "Moderation.create",
        ),
    ),
)
