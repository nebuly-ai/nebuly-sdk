from __future__ import annotations

import inspect
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Dict, Generator, List, cast
from uuid import UUID

from nebuly.entities import HistoryEntry, InteractionWatch, Observer, SpanWatch
from nebuly.exceptions import (
    AlreadyInInteractionContext,
    InteractionContextInitiationError,
    InteractionMustBeLocalVariable,
    MissingRequiredNebulyFieldError,
    NotInInteractionContext,
)


class InteractionContext:  # pylint: disable=too-many-instance-attributes
    def __init__(  # pylint: disable=too-many-arguments
        self,
        user: str | None = None,
        user_group_profile: str | None = None,
        initial_input: str | None = None,
        final_output: str | None = None,
        history: list[HistoryEntry] | None = None,
        hierarchy: dict[UUID, UUID | None] | None = None,
        spans: list[SpanWatch] | None = None,
        do_not_call_directly: bool = False,
        tags: dict[str, str] | None = None,
        feature_flags: list[str] | None = None,
        nebuly_api_key: str | None = None,
    ) -> None:
        if not do_not_call_directly:
            raise InteractionContextInitiationError(
                "Interaction cannot be directly instantiate, use the"
                " 'new_interaction' contextmanager"
            )
        self._observer: Observer | None = None
        self._finished: bool = False

        self.user = user
        self.user_group_profile = user_group_profile
        self.input = initial_input
        self.output = final_output
        self.spans = [] if spans is None else spans
        self.history = history
        self.hierarchy = hierarchy
        self.time_start = datetime.now(timezone.utc)
        self.tags = tags
        self.feature_flags = feature_flags
        self.nebuly_api_key = nebuly_api_key

    def set_input(self, value: str) -> None:
        self.input = value

    def set_history(self, value: list[dict[str, str]] | list[HistoryEntry]) -> None:
        if len(value) == 0:
            self.history = []
        elif isinstance(value[0], dict):
            value = cast(List[Dict[str, str]], value)
            history = [
                message
                for message in value
                if message["role"].lower() in ["user", "assistant"]
            ]
            if len(history) % 2 != 0:
                raise ValueError(
                    "Odd number of chat history elements, please provide "
                    "a valid history."
                )
            self.history = [
                HistoryEntry(
                    user=history[i]["content"], assistant=history[i + 1]["content"]
                )
                for i in range(len(history) - 1)
            ]
        else:
            value = cast(List[HistoryEntry], value)
            self.history = value

    def set_output(self, value: str) -> None:
        self.output = value

    def _set_observer(self, observer: Observer) -> None:
        self._observer = observer

    def _add_span(self, value: SpanWatch) -> None:
        self.spans.append(value)

    def _finish(self) -> None:
        self._finished = True
        self.hierarchy = {}
        for span in self.spans:
            if span.provider_extras is None:
                continue
            parent_id = span.provider_extras.get("parent_run_id")
            if parent_id is not None:
                self.hierarchy[span.span_id] = UUID(parent_id)
        if self._observer is not None:
            self._observer(self._as_interaction_watch())

    def _set_user(self, value: str) -> None:
        self.user = value

    def _set_user_group_profile(self, value: str | None) -> None:
        self.user_group_profile = value

    def _add_tags(self, tags: dict[str, str]) -> None:
        if self.tags is None:
            self.tags = {}
        self.tags = {**self.tags, **tags}

    def _add_feature_flags(self, flags: list[str]) -> None:
        if self.feature_flags is None:
            self.feature_flags = []
        self.feature_flags.extend(flags)

    def _set_api_key(self, api_key: str) -> None:
        self.nebuly_api_key = api_key

    def _validate_interaction(self) -> None:
        if self.input is None:
            raise ValueError("Interaction has no input.")
        if self.output is None:
            raise ValueError("Interaction has no output.")
        if self.history is None:
            raise ValueError("Interaction has no history.")
        if self.hierarchy is None:
            raise ValueError("Interaction has no hierarchy.")
        if self.hierarchy:
            for value in self.hierarchy.values():
                if value is not None and value not in self.hierarchy:
                    raise ValueError(
                        f"Interaction hierarchy has a reference to a non-existing "
                        f"event: {value}"
                    )
        if self.user is None:
            raise MissingRequiredNebulyFieldError(
                "Missing required nebuly field: 'user_id'. Please add it when calling "
                "the original provider method."
            )

    def _as_interaction_watch(self) -> InteractionWatch:
        self._validate_interaction()
        return InteractionWatch(
            input=self.input,  # type: ignore
            output=self.output,  # type: ignore
            time_start=self.time_start,
            time_end=datetime.now(timezone.utc),
            spans=self.spans,
            history=self.history,  # type: ignore
            hierarchy=self.hierarchy,  # type: ignore
            end_user=self.user,  # type: ignore
            end_user_group_profile=self.user_group_profile,
            tags=self.tags,
            feature_flags=self.feature_flags,
            api_key=self.nebuly_api_key,
        )


def get_nearest_open_interaction() -> InteractionContext:
    frames = inspect.stack()
    for frame in frames[::-1]:
        for v in frame.frame.f_locals.values():
            if (
                isinstance(v, InteractionContext)
                and not v._finished  # pylint: disable=protected-access
            ):
                return v
    raise NotInInteractionContext()


@contextmanager
def new_interaction(
    user_id: str | None = None,
    user_group_profile: str | None = None,
    nebuly_tags: dict[str, str] | None = None,
    auto_publish: bool = True,
    feature_flags: list[str] | None = None,
) -> Generator[InteractionContext, None, None]:
    try:
        get_nearest_open_interaction()
        raise AlreadyInInteractionContext()
    except NotInInteractionContext:
        pass

    try:
        yield InteractionContext(
            user_id,
            user_group_profile,
            tags=nebuly_tags,
            feature_flags=feature_flags,
            do_not_call_directly=True,
        )
    finally:
        if auto_publish:
            try:
                interaction = get_nearest_open_interaction()
            except NotInInteractionContext:
                raise InteractionMustBeLocalVariable()  # pylint: disable=raise-missing-from  # noqa: E501
            interaction._finish()  # pylint: disable=protected-access
