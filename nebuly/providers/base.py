from __future__ import annotations

import abc
from typing import Any

from nebuly.entities import ModelInput


class ProviderDataExtractor(abc.ABC):
    @abc.abstractmethod
    def extract_input_and_history(self) -> ModelInput | list[ModelInput]:
        raise NotImplementedError("extract_input_and_history not implemented")

    @abc.abstractmethod
    def extract_output(self, stream: bool, outputs: Any) -> str | list[str]:
        raise NotImplementedError("extract_output not implemented")
