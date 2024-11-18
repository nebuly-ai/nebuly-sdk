# pylint: disable=duplicate-code
from __future__ import annotations

from typing import Any, List, cast

from huggingface_hub.inference._text_generation import (  # type: ignore
    TextGenerationResponse,
    TextGenerationStreamResponse,
)
from huggingface_hub.inference._types import ConversationalOutput  # type: ignore

from nebuly.entities import HistoryEntry, ModelInput
from nebuly.providers.base import ProviderDataExtractor
from nebuly.providers.utils import get_argument


class HuggingFaceHubDataExtractor(ProviderDataExtractor):
    def __init__(
        self,
        function_name: str,
        original_args: tuple[Any, ...],
        original_kwargs: dict[str, Any],
    ):
        self.function_name = function_name
        self.original_args = original_args
        self.original_kwargs = original_kwargs

    def _extract_history(self) -> list[HistoryEntry]:
        generated_responses = get_argument(
            self.original_args, self.original_kwargs, "generated_responses", 2
        )
        past_user_inputs = get_argument(
            self.original_args, self.original_kwargs, "past_user_inputs", 3
        )
        history = []

        for user_input, assistant_response in zip(
            past_user_inputs if past_user_inputs is not None else [],
            generated_responses if generated_responses is not None else [],
        ):
            history.append(HistoryEntry(user=user_input, assistant=assistant_response))
        return history

    def extract_input_and_history(self, outputs: Any) -> ModelInput:
        if self.function_name in [
            "InferenceClient.conversational",
            "AsyncInferenceClient.conversational",
        ]:
            prompt = get_argument(self.original_args, self.original_kwargs, "text", 1)
            history = self._extract_history()
            return ModelInput(prompt=prompt, history=history)
        if self.function_name in [
            "InferenceClient.text_generation",
            "AsyncInferenceClient.text_generation",
        ]:
            prompt = get_argument(self.original_args, self.original_kwargs, "prompt", 1)
            return ModelInput(prompt=prompt)

        raise ValueError(f"Unknown function name: {self.function_name}")

    def extract_output(
        self,
        stream: bool,
        outputs: (
            str
            | ConversationalOutput
            | TextGenerationResponse
            | list[str]
            | list[TextGenerationStreamResponse]
        ),
    ) -> str:
        if stream and isinstance(outputs, list):
            return self._extract_output_generator(outputs)

        if self.function_name in [  # ConversationalOutput
            "InferenceClient.conversational",
            "AsyncInferenceClient.conversational",
        ]:
            result = cast(ConversationalOutput, outputs)
            return cast(str, result["generated_text"])
        if isinstance(
            outputs, (str, TextGenerationResponse)
        ) and self.function_name in [
            "InferenceClient.text_generation",
            "AsyncInferenceClient.text_generation",
        ]:
            if isinstance(outputs, TextGenerationResponse):
                return cast(str, outputs.generated_text)
            return outputs

        raise ValueError(
            f"Unknown function name: {self.function_name} or "
            f"output type: {type(outputs)}"
        )

    def _extract_output_generator(
        self, outputs: list[str] | list[TextGenerationStreamResponse]
    ) -> str:
        if self.function_name in [
            "InferenceClient.text_generation",
            "AsyncInferenceClient.text_generation",
        ]:
            if all(isinstance(output, str) for output in outputs):
                result = cast(List[str], outputs)
                return "".join(result)
            if all(
                isinstance(output, TextGenerationStreamResponse) for output in outputs
            ):
                result = cast(List[TextGenerationStreamResponse], outputs)
                return "".join(
                    [
                        (
                            output.generated_text  # type: ignore
                            if output.generated_text is not None  # type: ignore
                            else ""
                        )
                        for output in result
                    ]
                )

        raise ValueError(
            f"Unknown function name: {self.function_name} or "
            f"output type: {type(outputs)}"
        )
