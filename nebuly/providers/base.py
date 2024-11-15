from __future__ import annotations

import abc
import copyreg
from typing import Any, Callable

from nebuly.entities import ModelInput


class ProviderDataExtractor(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self,
        original_args: tuple[Any, ...],
        original_kwargs: dict[str, Any],
        function_name: str,
    ):
        raise NotImplementedError("init not implemented")

    @abc.abstractmethod
    def extract_input_and_history(self, outputs: Any) -> ModelInput | list[ModelInput]:
        raise NotImplementedError("extract_input_and_history not implemented")

    @abc.abstractmethod
    def extract_output(self, stream: bool, outputs: Any) -> str | list[str]:
        raise NotImplementedError("extract_output not implemented")

    def extract_media(self) -> list[str] | None:
        return None


class PicklerHandler(abc.ABC):
    @property
    @abc.abstractmethod
    def _object_key_attribute_mapping(
        self,
    ) -> dict[Callable[[Any], Any], dict[str, list[str]]]:
        raise NotImplementedError("This method should be implemented by subclasses")

    def _get_attribute(self, obj: Any, attribute: str) -> Any:
        if "." not in attribute:
            return getattr(obj, attribute)

        attributes = attribute.split(".")
        return self._get_attribute(
            getattr(obj, attributes[0]), ".".join(attributes[1:])
        )

    @staticmethod
    def _unpickle_object(
        constructor: Callable[[Any], Any], kwargs: dict[str, Any]
    ) -> Any:
        return constructor(**kwargs)  # type: ignore

    def _pickle_object(self, obj: Any) -> tuple[
        Callable[[Callable[[Any], Any], dict[str, Any]], Any],
        tuple[Callable[[Any], Any], dict[str, Any]],
    ]:
        return self._unpickle_object, (
            obj.__class__,
            {
                key: self._get_attribute(obj, attribute)
                for key, attribute in zip(
                    self._object_key_attribute_mapping[obj.__class__]["key_names"],
                    self._object_key_attribute_mapping[obj.__class__][
                        "attribute_names"
                    ],
                )
            },
        )

    def handle_unpickable_object(self, obj: Callable[[Any], Any]) -> None:
        copyreg.pickle(obj, self._pickle_object)  # type: ignore
