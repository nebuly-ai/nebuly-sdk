from __future__ import annotations

from inspect import isasyncgenfunction
from typing import Any, Callable, cast
from uuid import UUID

from langchain.callbacks.manager import CallbackManager
from langchain.chains.base import Chain
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.prompts.chat import (
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from nebuly.contextmanager import (
    NotInInteractionContext,
    get_nearest_open_interaction,
    new_interaction,
)
from nebuly.tracking_handlers import LangChainTrackingHandler


def _get_tracking_info_for_provider_call(**kwargs: Any) -> dict[str, Any]:
    """
    This function is a hack to add the chain run_ids info to the kwargs of the
    provider call. This is needed to associate each call to a provider (ex OpenAI)
    with the belonging chain.
    """
    callbacks = kwargs.get("callbacks")
    if not isinstance(callbacks, CallbackManager) or callbacks.parent_run_id is None:
        # If the llm/chat_model is called without a CallbackManager, it's not
        # called from a chain, so we don't need to add the run ids info
        return {}

    # Get the parent_run_id from the CallbackManager
    callback_manager = callbacks
    parent_run_id = cast(UUID, callback_manager.parent_run_id)
    additional_kwargs = {"nebuly_parent_run_id": parent_run_id}

    # Get the root_run_id from the LangChainTrackingHandler
    for handler in callback_manager.handlers:
        if isinstance(handler, LangChainTrackingHandler):
            interaction = get_nearest_open_interaction()
            root_run_id = interaction._events_storage.get_root_id(parent_run_id)
            additional_kwargs["nebuly_root_run_id"] = root_run_id
            break

    return additional_kwargs


def _get_input_and_history(
    chain: Chain, inputs: dict[str, Any] | Any
) -> tuple[str | None, list[tuple[str, Any]] | None]:
    if isinstance(inputs, str):
        return inputs, None

    chains = getattr(chain, "chains", None)
    if chains is not None:
        # If the chain is a SequentialChain, we need to get the
        # prompt from the first chain
        prompt = getattr(chains[0], "prompt", None)
    else:
        # If the chain is not a SequentialChain, we need to get
        # the prompt from the chain
        prompt = getattr(chain, "prompt", None)
    if "prompt" in inputs:
        prompt = inputs["prompt"]

    if not isinstance(inputs, dict) or prompt is None:
        return None, None

    if isinstance(prompt, str):
        return prompt, None

    if isinstance(prompt, PromptTemplate):
        return (
            prompt.format(**{key: inputs.get(key) for key in prompt.input_variables}),
            None,
        )

    if isinstance(prompt, ChatPromptTemplate):
        messages = []
        for message in prompt.messages:
            input_vars = {key: inputs.get(key) for key in message.input_variables}
            if isinstance(message, SystemMessagePromptTemplate):
                messages.append(("system", message.format(**input_vars).content))
            elif isinstance(message, HumanMessagePromptTemplate):
                messages.append(("human", message.format(**input_vars).content))
            elif isinstance(message, AIMessagePromptTemplate):
                messages.append(("ai", message.format(**input_vars).content))
        if len(messages) > 0:
            if len(messages) == 1:
                return messages[0][1], None
            return messages[-1][1], messages[:-1]


def wrap_langchain(
    observer: Callable[[Any], None],
    function_name: str,
    f: Callable[[Any], Any],
    args: tuple[Any],
    kwargs: dict[str, Any],
):
    if function_name.startswith(
        ("llms.base.BaseLLM", "chat_models.base.BaseChatModel")
    ):
        additional_kwargs = _get_tracking_info_for_provider_call(**kwargs)
        return f(*args, **kwargs, **additional_kwargs)
    if function_name.startswith("chains.base.Chain"):
        try:
            get_nearest_open_interaction()
            return f(*args, **kwargs)
        except NotInInteractionContext:
            with new_interaction() as interaction:
                inputs = kwargs.get("inputs")
                handler = [
                    handler
                    for handler in kwargs.get("callbacks")
                    if isinstance(handler, LangChainTrackingHandler)
                ][0]
                interaction._set_user(handler.nebuly_user)
                interaction._set_user_group_profile(handler.nebuly_user_group)
                interaction._set_observer(observer)
                chain_input, history = _get_input_and_history(args[0], inputs)
                interaction.set_input(chain_input)
                interaction.set_history(history)
                original_res = f(*args, **kwargs)
                interaction.set_output(original_res)
                return original_res


async def wrap_langchain_async(
    observer: Callable[[Any], None],
    function_name: str,
    f: Callable[[Any], Any],
    args: tuple[Any],
    kwargs: dict[str, Any],
):
    if function_name.startswith(
        ("llms.base.BaseLLM", "chat_models.base.BaseChatModel")
    ):
        additional_kwargs = _get_tracking_info_for_provider_call(**kwargs)
        if isasyncgenfunction(f):
            return f(*args, **kwargs, **additional_kwargs)
        return await f(*args, **kwargs, **additional_kwargs)
    if function_name.startswith("chains.base.Chain"):
        try:
            get_nearest_open_interaction()
            if isasyncgenfunction(f):
                return f(*args, **kwargs)
            return await f(*args, **kwargs)
        except NotInInteractionContext:
            with new_interaction() as interaction:
                inputs = kwargs.get("inputs")
                if isinstance(inputs, dict):
                    user = inputs.pop("nebuly_user", None)
                    user_group = inputs.pop("nebuly_user_group_profile", None)
                    interaction._set_user(user)
                    interaction._set_user_group_profile(user_group)
                interaction._set_observer(observer)
                chain_input, history = _get_input_and_history(args[0], inputs)
                interaction.set_input(chain_input)
                interaction.set_history(history)
                if isasyncgenfunction(f):
                    original_res = f(*args, **kwargs)
                else:
                    original_res = await f(*args, **kwargs)
                interaction.set_output(original_res)
                return original_res


# TODO: Add test for sequential chains
