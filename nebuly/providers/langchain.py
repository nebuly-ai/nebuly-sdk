from __future__ import annotations

import logging
from copy import deepcopy
from inspect import isasyncgenfunction
from typing import Any, AsyncGenerator, Callable, Generator, cast
from uuid import UUID

from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import AsyncCallbackManager, CallbackManager
from langchain.chains.base import Chain
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.prompts.chat import AIMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema.messages import AIMessage
from langchain.schema.runnable.base import RunnableSequence

from nebuly.contextmanager import (
    InteractionContext,
    get_nearest_open_interaction,
    new_interaction,
)
from nebuly.entities import HistoryEntry, ModelInput, Observer
from nebuly.exceptions import MissingRequiredNebulyFieldError, NotInInteractionContext
from nebuly.providers.utils import get_argument
from nebuly.tracking_handlers import LangChainTrackingHandler

logger = logging.getLogger(__name__)


def _get_tracking_info_for_provider_call(**kwargs: Any) -> dict[str, Any]:
    """
    This function is a hack to add the chain run_ids info to the kwargs of the
    provider call. This is needed to associate each call to a provider (ex OpenAI)
    with the belonging chain.
    """
    callbacks = kwargs.get("callbacks")
    if (
        not isinstance(callbacks, (CallbackManager, AsyncCallbackManager))
        or callbacks.parent_run_id is None
    ):
        # If the llm/chat_model is called without a CallbackManager, it's not
        # called from a chain, so we don't need to add the run ids info
        return {}

    # Get the parent_run_id from the CallbackManager
    callback_manager = callbacks
    parent_run_id = cast(UUID, callback_manager.parent_run_id)
    additional_kwargs: dict[str, Any] = {"parent_run_id": str(parent_run_id)}

    # Get the root_run_id from the LangChainTrackingHandler
    for handler in callback_manager.handlers:
        if isinstance(handler, LangChainTrackingHandler):
            interaction = handler.current_interaction
            root_run_id = interaction._events_storage.get_root_id(  # pylint: disable=protected-access  # noqa: E501
                parent_run_id
            )
            additional_kwargs["root_run_id"] = str(root_run_id)
            additional_kwargs["nebuly_interaction"] = interaction
            break

    return additional_kwargs


def _process_prompt_template(
    inputs: dict[str, Any] | Any, prompt: PromptTemplate
) -> str:
    if isinstance(inputs, dict):
        return prompt.format(**{key: inputs.get(key) for key in prompt.input_variables})
    return prompt.format(**{prompt.input_variables[0]: inputs})


def _process_chat_prompt_template(
    inputs: dict[str, Any] | Any, prompt: ChatPromptTemplate
) -> tuple[str, list[HistoryEntry]]:
    messages = []
    for message in prompt.messages:
        if isinstance(inputs, dict):
            input_vars = {key: inputs.get(key) for key in message.input_variables}  # type: ignore  # noqa: E501  # pylint: disable=line-too-long
        else:
            input_vars = {message.input_variables[0]: inputs}  # type: ignore
        if isinstance(message, (HumanMessagePromptTemplate, AIMessagePromptTemplate)):
            messages.append((message.format(**input_vars).content))
    last_prompt = messages[-1]
    message_history = messages[:-1]

    if len(message_history) % 2 != 0:
        logger.warning("Odd number of chat history elements, ignoring last element")
        message_history = message_history[:-1]

    # Convert the history to [(user, assistant), ...] format
    history = [
        HistoryEntry(user=message_history[i], assistant=message_history[i + 1])
        for i in range(0, len(message_history), 2)
        if i < len(message_history) - 1
    ]

    return last_prompt, history


def _get_input_and_history(chain: Chain, inputs: dict[str, Any] | Any) -> ModelInput:
    chains = getattr(chain, "chains", None)
    if chains is not None:
        # If the chain is a SequentialChain, we need to get the
        # prompt from the first chain
        prompt = getattr(chains[0], "prompt", None)
    else:
        # If the chain is not a SequentialChain, we need to get
        # the prompt from the chain
        prompt = getattr(chain, "prompt", None)
        if prompt is None:
            if not isinstance(inputs, dict):
                return ModelInput(prompt=inputs)
            return ModelInput(prompt=inputs["input"])

    if isinstance(prompt, PromptTemplate):
        return ModelInput(prompt=_process_prompt_template(inputs, prompt))

    if isinstance(prompt, ChatPromptTemplate):
        prompt, history = _process_chat_prompt_template(inputs, prompt)
        return ModelInput(prompt=prompt, history=history)

    raise ValueError(f"Unknown prompt type: {prompt}")


def _get_input_and_history_runnable_seq(
    sequence: RunnableSequence[Any, Any], inputs: dict[str, Any] | Any
) -> ModelInput:
    first = getattr(sequence, "first", None)

    if isinstance(first, PromptTemplate):
        return ModelInput(prompt=_process_prompt_template(inputs, first))

    if isinstance(first, ChatPromptTemplate):
        prompt, history = _process_chat_prompt_template(inputs, first)
        return ModelInput(prompt=prompt, history=history)

    return ModelInput(prompt="")


def _get_output_chain(chain: Chain, result: dict[str, Any]) -> str:
    if len(chain.output_keys) == 1:
        return str(result[chain.output_keys[0]])
    output = {}
    for key in chain.output_keys:
        output[key] = result[key]
    return _parse_output(output)


def _parse_output(output: str | dict[str, Any] | AIMessage) -> str:
    if isinstance(output, dict):
        return "\n".join([f"{key}: {value}" for key, value in output.items()])
    if isinstance(output, AIMessage):
        return output.content
    return output


def _get_tracking_handler(
    callbacks: list[BaseCallbackHandler] | CallbackManager,
) -> LangChainTrackingHandler:
    if isinstance(callbacks, CallbackManager):
        handlers = callbacks.handlers
    else:
        handlers = callbacks

    for handler in handlers:
        if isinstance(handler, LangChainTrackingHandler):
            return handler
    raise ValueError("LangChainTrackingHandler not found")


def _result_from_generator(
    generator: Generator[Any, Any, Any], interaction: InteractionContext
) -> Generator[Any, Any, Any]:
    original_results = []

    for element in generator:
        logger.debug("Yielding %s", element)
        original_results.append(deepcopy(str(element.content)))
        yield element

    original_result = "".join(original_results)

    interaction.set_output(_parse_output(original_result))
    interaction._finish()  # pylint: disable=protected-access


async def _result_from_generator_async(
    generator: AsyncGenerator[Any, Any], interaction: InteractionContext
) -> AsyncGenerator[Any, Any]:
    original_results = []

    async for element in generator:
        logger.debug("Yielding %s", element)
        original_results.append(deepcopy(element.content))
        yield element

    original_result = "".join(original_results)
    interaction.set_output(_parse_output(original_result))
    interaction._finish()  # pylint: disable=protected-access


def wrap_langchain(  # pylint: disable=too-many-return-statements,too-many-statements
    observer: Observer,
    function_name: str,
    f: Callable[[Any], Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
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
            handler = _get_tracking_handler(kwargs.get("callbacks", []))
            with new_interaction(
                user_id=handler.nebuly_user,
                user_group_profile=handler.nebuly_user_group,
            ) as interaction:
                inputs = kwargs.get("inputs")
                interaction._set_observer(observer)  # pylint: disable=protected-access
                model_input = _get_input_and_history(args[0], inputs)
                interaction.set_input(model_input.prompt)
                interaction.set_history(model_input.history)
                original_res = f(*args, **kwargs)
                interaction.set_output(_get_output_chain(args[0], original_res))
                return original_res

    if function_name.startswith("indexes.vectorstore.VectorStoreIndexWrapper"):
        try:
            get_nearest_open_interaction()
            return f(*args, **kwargs)
        except NotInInteractionContext:
            try:
                user_id = kwargs.pop("user_id")
            except KeyError as e:
                raise MissingRequiredNebulyFieldError("user_id") from e
            with new_interaction(
                user_id=user_id,
                user_group_profile=kwargs.pop("user_group_profile", None),
            ) as interaction:
                inputs = get_argument(args, kwargs, "question", 1)
                interaction._set_observer(observer)  # pylint: disable=protected-access
                model_input = _get_input_and_history(args[0], inputs)
                interaction.set_input(model_input.prompt)
                interaction.set_history(model_input.history)
                original_res = f(*args, **kwargs)
                interaction.set_output(_parse_output(original_res))
                return original_res

    if function_name.startswith("schema.runnable.base"):
        config = get_argument(args, kwargs, "config", 2)
        handlers = config.get("callbacks", [])
        if isinstance(handlers, CallbackManager):
            # We are in an inner RunnableSequence, we could be in a thread so
            # we need to get interaction from the TrackingHandler
            handler = _get_tracking_handler(handlers)
            interaction = handler.current_interaction
            return f(*args, **kwargs)

        # Outer RunnableSequence, we are in the main thread
        try:
            get_nearest_open_interaction()
            return f(*args, **kwargs)
        except NotInInteractionContext:
            handler = _get_tracking_handler(handlers)
            with new_interaction(
                user_id=handler.nebuly_user,
                user_group_profile=handler.nebuly_user_group,
                auto_publish=False,
            ) as interaction:
                inputs = get_argument(args, kwargs, "input", 1)
                model_input = _get_input_and_history_runnable_seq(args[0], inputs)
                interaction._set_observer(observer)  # pylint: disable=protected-access
                if model_input.prompt != "":
                    interaction.set_input(model_input.prompt)
                    interaction.set_history(model_input.history)
                original_res = f(*args, **kwargs)
                if isinstance(original_res, Generator):
                    logger.debug("Result is a generator")
                    return _result_from_generator(original_res, interaction)
                interaction.set_output(_parse_output(original_res))
                interaction._finish()  # pylint: disable=protected-access
                return original_res

    raise ValueError(f"Unknown function name: {function_name}")


async def wrap_langchain_async(  # pylint: disable=too-many-return-statements,too-many-statements, too-many-branches  # noqa: E501
    observer: Observer,
    function_name: str,
    f: Callable[[Any], Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
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
                    user = inputs.pop("user_id", None)
                    user_group = inputs.pop("user_group_profile", None)
                    interaction._set_user(user)  # pylint: disable=protected-access
                    interaction._set_user_group_profile(  # pylint: disable=protected-access  # noqa: E501
                        user_group
                    )
                interaction._set_observer(observer)  # pylint: disable=protected-access
                model_input = _get_input_and_history(args[0], inputs)
                interaction.set_input(model_input.prompt)
                interaction.set_history(model_input.history)
                if isasyncgenfunction(f):
                    original_res = f(*args, **kwargs)
                else:
                    original_res = await f(*args, **kwargs)
                interaction.set_output(original_res)
                return original_res

    if function_name.startswith("schema.runnable.base"):
        config = get_argument(args, kwargs, "config", 2)
        handlers = config.get("callbacks", [])
        if isinstance(handlers, CallbackManager):
            # We are in an inner RunnableSequence, we could be in a thread so
            # we need to get interaction from the TrackingHandler
            handler = _get_tracking_handler(handlers)
            interaction = handler.current_interaction
            return await f(*args, **kwargs)

        # Outer RunnableSequence, we are in the main thread
        try:
            get_nearest_open_interaction()
            if isasyncgenfunction(f):
                return f(*args, **kwargs)
            return await f(*args, **kwargs)
        except NotInInteractionContext:
            handler = _get_tracking_handler(handlers)
            with new_interaction(
                user_id=handler.nebuly_user,
                user_group_profile=handler.nebuly_user_group,
                auto_publish=False,
            ) as interaction:
                handler.set_interaction(interaction)
                inputs = get_argument(args, kwargs, "input", 1)
                model_input = _get_input_and_history_runnable_seq(args[0], inputs)
                interaction._set_observer(observer)  # pylint: disable=protected-access
                if model_input.prompt != "":
                    interaction.set_input(model_input.prompt)
                    interaction.set_history(model_input.history)
                if isasyncgenfunction(f):
                    original_res = f(*args, **kwargs)
                else:
                    original_res = await f(*args, **kwargs)

                if isinstance(original_res, AsyncGenerator):
                    logger.debug("Result is a generator")
                    return _result_from_generator_async(original_res, interaction)
                interaction.set_output(_parse_output(original_res))
                interaction._finish()  # pylint: disable=protected-access
                return original_res

    raise ValueError(f"Unknown function name: {function_name}")
