from datetime import datetime, timezone

from nebuly.entities import DevelopmentPhase, Watched


def test_watched_to_dict():
    watched = Watched(
        module="module",
        version="version",
        function="function",
        called_start=datetime.now(tz=timezone.utc),
        called_end=datetime.now(tz=timezone.utc),
        called_with_args=("arg1", "arg2"),
        called_with_kwargs={"kwarg1": "kwarg1", "kwarg2": "kwarg2"},
        called_with_nebuly_kwargs={
            "nebuly_phase": DevelopmentPhase.EXPERIMENTATION,
            "nebuly_project": "nebuly_project",
        },
        returned="returned",
        generator=False,
        generator_first_element_timestamp=None,
        provider_extras={"provider_extra": "provider_extra"},
    )
    assert watched.to_dict() == {
        "module": "module",
        "version": "version",
        "function": "function",
        "called_start": watched.called_start.isoformat(),
        "called_end": watched.called_end.isoformat(),
        "called_with_args": ("arg1", "arg2"),
        "called_with_kwargs": {"kwarg1": "kwarg1", "kwarg2": "kwarg2"},
        "called_with_nebuly_kwargs": {
            "nebuly_phase": "experimentation",
            "nebuly_project": "nebuly_project",
        },
        "returned": "returned",
        "generator": False,
        "generator_first_element_timestamp": None,
        "provider_extras": {"provider_extra": "provider_extra"},
    }
