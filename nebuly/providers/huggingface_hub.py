from typing import Any

from huggingface_hub.inference._types import ConversationalOutput

from nebuly.providers.utils import get_argument


def extract_hf_hub_input_and_history(
    original_args: tuple[Any],
    original_kwargs: dict[str, Any],
    function_name: str,
) -> tuple[str, list[tuple[str, Any]]]:
    if function_name == "InferenceClient.conversational":
        prompt = get_argument(original_args, original_kwargs, "text", 1)
        generated_responses = get_argument(
            original_args, original_kwargs, "generated_responses", 2
        )
        past_user_inputs = get_argument(
            original_args, original_kwargs, "past_user_inputs", 3
        )
        history = []
        for user_input, assistant_response in zip(
            past_user_inputs if past_user_inputs is not None else [],
            generated_responses if generated_responses is not None else [],
        ):
            history.append(("user", user_input))
            history.append(("assistant", assistant_response))
        return prompt, history


def extract_hf_hub_output(function_name: str, output: ConversationalOutput) -> str:
    if function_name == "InferenceClient.conversational":
        return output["generated_text"]
