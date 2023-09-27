from __future__ import annotations

import copyreg
import importlib
import logging
import sys
from copy import deepcopy
from datetime import datetime, timezone
from functools import wraps
from inspect import isasyncgenfunction, iscoroutinefunction
from types import ModuleType
from typing import Any, AsyncGenerator, Callable, Generator, Iterable

from anthropic import AsyncStream, Stream
from cohere.responses.chat import StreamingChat
from cohere.responses.generation import StreamingGenerations

from nebuly.contextmanager import (
    NotInInteractionContext,
    get_nearest_open_interaction,
    new_interaction,
)
from nebuly.entities import Observer, Package, SpanWatch

logger = logging.getLogger(__name__)


AsyncGen = AsyncGenerator | StreamingGenerations | StreamingChat | AsyncStream
SyncGen = Generator | StreamingGenerations | StreamingChat | Stream


def check_no_packages_already_imported(packages: Iterable[Package]) -> None:
    """
    Check that no packages in packages have already been imported.
    """
    for package in packages:
        if package.name in sys.modules:
            logger.warning("%s already imported", package.name)


def import_and_patch_packages(packages: Iterable[Package], observer: Observer) -> None:
    """
    Import each package in packages and patch it with the _observer.
    """
    for package in packages:
        try:
            _monkey_patch(package, observer)
        except ImportError:
            pass


def _monkey_patch(package: Package, observer: Observer) -> None:
    module = importlib.import_module(package.name)
    for attr in package.to_patch:
        try:
            _monkey_patch_attribute(attr, module, observer)
        except (AttributeError, ImportError):
            logger.warning("Failed to patch %s", attr)


def _monkey_patch_attribute(attr: str, module: ModuleType, observer: Observer) -> None:
    version = module.__version__ if hasattr(module, "__version__") else "unknown"
    tmp_component = module
    path = attr.split(".")
    for i, component_name in enumerate(path[:-1]):
        try:
            tmp_component = getattr(tmp_component, component_name)
        except AttributeError:
            # Handle the case when the module is imported at runtime
            tmp_component = importlib.import_module(
                f"{module.__name__}.{'.'.join(path[:i+1])}"
            )

    setattr(
        tmp_component,
        path[-1],
        _patcher(observer, module.__name__, version, attr)(
            getattr(tmp_component, path[-1])
        ),
    )


def _split_nebuly_kwargs(
    kwargs: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Split nebuly kwargs from function kwargs
    """

    nebuly_kwargs = {}
    function_kwargs = {}
    for key in kwargs:
        if key.startswith("nebuly_"):
            nebuly_kwargs[key] = kwargs[key]
        else:
            function_kwargs[key] = kwargs[key]
    return nebuly_kwargs, function_kwargs


def _extract_output(output: Any, module: str, function_name: str) -> str:
    if module == "openai":
        if function_name in ["Completion.create", "Completion.acreate"]:
            return output["choices"][0]["text"]
        if function_name in ["ChatCompletion.create", "ChatCompletion.acreate"]:
            return output["choices"][0]["message"]["content"]
    if module == "cohere":
        if function_name in ["Client.generate", "AsyncClient.generate"]:
            return output.generations[0].text
        if function_name in ["Client.chat", "AsyncClient.chat"]:
            return output.text
    if module == "anthropic":
        if function_name in [
            "resources.Completions.create",
            "resources.AsyncCompletions.create",
        ]:
            return output.completion
    if module == "huggingface_hub":
        if function_name == "InferenceClient.conversational":
            return output["generated_text"]
    if module == "google":
        if function_name == "generativeai.generate_text":
            return output.result
        if function_name in [
            "generativeai.chat",
            "generativeai.chat_async",
            "generativeai.discuss.ChatResponse.reply",
        ]:
            return output.messages[-1]["content"]
    if module == "vertexai":
        if function_name in [
            "language_models.TextGenerationModel.predict",
            "language_models.TextGenerationModel.predict_async",
            "language_models.ChatSession.send_message",
            "language_models.ChatSession.send_message_async",
        ]:
            return output.text
    return str(output)


def _extract_output_generator(
    outputs: list[Any], module: str, function_name: str
) -> str:
    if module == "openai":
        if function_name in ["Completion.create", "Completion.acreate"]:
            return "".join([output["choices"][0].get("text", "") for output in outputs])
        if function_name in ["ChatCompletion.create", "ChatCompletion.acreate"]:
            return "".join(
                [output["choices"][0]["delta"].get("content", "") for output in outputs]
            )
    if module == "cohere":
        if function_name in ["Client.generate", "AsyncClient.generate"]:
            return "".join([output.text for output in outputs])
        if function_name in ["Client.chat", "AsyncClient.chat"]:
            return "".join(
                [
                    output.text
                    for output in outputs
                    if output.event_type == "text-generation"
                ]
            )
    if module == "anthropic":
        if function_name in [
            "resources.Completions.create",
            "resources.AsyncCompletions.create",
        ]:
            return "".join([output.completion for output in outputs])
    if module == "vertexai":
        if function_name in [
            "language_models.TextGenerationModel.predict_streaming",
            "language_models.ChatSession.send_message_streaming",
        ]:
            return "".join([output.text for output in outputs])


def _add_interaction_span(  # pylint: disable=too-many-arguments
    original_args: tuple[Any, ...],
    original_kwargs: dict[str, Any],
    module: str,
    function_name: str,
    output: Any,
    observer: Observer,
    watched: SpanWatch,
    nebuly_kwargs: dict[str, Any],
    stream: bool = False,
) -> None:
    try:
        interaction = get_nearest_open_interaction()
        interaction._set_observer(observer)
        interaction._add_span(watched)
    except NotInInteractionContext:
        try:
            user_input, history = _extract_input_and_history(
                original_args, original_kwargs, module, function_name
            )
        except Exception:
            user_input, history = None, None
        with new_interaction() as interaction:
            interaction.set_input(user_input)
            interaction.set_history(history)
            interaction._set_observer(observer)
            interaction._add_span(watched)
            interaction._set_user(nebuly_kwargs.get("nebuly_user"))
            interaction._set_user_group_profile(
                nebuly_kwargs.get("nebuly_user_group_profile")
            )
            interaction.set_output(
                _extract_output(output, module, function_name)
                if not stream
                else _extract_output_generator(output, module, function_name)
            )


def _get_argument(
    args: tuple[Any, ...], kwargs: dict[str, Any], arg_name: str, arg_idx: int
) -> Any:
    """
    Get the argument both when it's passed as a positional argument or as a
    keyword argument
    """
    if kwargs.get(arg_name) is not None:
        return kwargs.get(arg_name)

    if len(args) > arg_idx:
        return args[arg_idx]

    return None


def _extract_input_and_history(
    original_args: tuple[Any, ...],
    original_kwargs: dict[str, Any],
    module: str,
    function_name: str,
) -> tuple[str, list[tuple[str, Any]]]:
    if module == "openai":
        if function_name in ["Completion.create", "Completion.acreate"]:
            return original_kwargs.get("prompt"), []
        if function_name in ["ChatCompletion.create", "ChatCompletion.acreate"]:
            history = [
                (el["role"], el["content"])
                for el in original_kwargs.get("messages")[:-1]
                if len(original_kwargs.get("messages", [])) > 1
            ]
            return original_kwargs.get("messages")[-1]["content"], history
    if module == "cohere":
        if function_name in ["Client.generate", "AsyncClient.generate"]:
            prompt = _get_argument(original_args, original_kwargs, "prompt", 1)
            return prompt, []
        if function_name in ["Client.chat", "AsyncClient.chat"]:
            prompt = _get_argument(original_args, original_kwargs, "message", 1)
            chat_history = _get_argument(
                original_args, original_kwargs, "chat_history", 7
            )
            history = [(el["user_name"], el["message"]) for el in chat_history]
            return prompt, history
    if module == "anthropic":
        if function_name in [
            "resources.Completions.create",
            "resources.AsyncCompletions.create",
        ]:
            return original_kwargs.get("prompt"), []
    if module == "huggingface_hub":
        if function_name == "InferenceClient.conversational":
            prompt = _get_argument(original_args, original_kwargs, "text", 1)
            generated_responses = _get_argument(
                original_args, original_kwargs, "generated_responses", 2
            )
            past_user_inputs = _get_argument(
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
    if module == "google":
        if function_name == "generativeai.generate_text":
            return original_kwargs.get("prompt"), []
        if function_name in ["generativeai.chat", "generativeai.chat_async"]:
            history = [
                ("user" if i % 2 == 0 else "assistant", el)
                for i, el in enumerate(original_kwargs.get("messages")[:-1])
                if len(original_kwargs.get("messages")) > 1
            ]
            return original_kwargs.get("messages")[-1], history
        if function_name == "generativeai.discuss.ChatResponse.reply":
            prompt = _get_argument(original_args, original_kwargs, "message", 1)
            history = [
                ("user" if el["author"] == "0" else "assistant", el["content"])
                for el in getattr(original_args[0], "messages", [])
            ]
            return prompt, history
    if module == "vertexai":
        if function_name in [
            "language_models.TextGenerationModel.predict",
            "language_models.TextGenerationModel.predict_async",
            "language_models.TextGenerationModel.predict_streaming",
            "language_models.ChatSession.send_message",
            "language_models.ChatSession.send_message_async",
            "language_models.ChatSession.send_message_streaming",
        ]:
            prompt = _get_argument(
                args=original_args,
                kwargs=original_kwargs,
                arg_name="prompt"
                if "TextGenerationModel" in function_name
                else "message",
                arg_idx=1,
            )
            history = [
                ("user" if el.author == "user" else "assistant", el.content)
                for el in getattr(original_args[0], "message_history", [])
            ]
            return prompt, history


def watch_from_generator(  # pylint: disable=too-many-arguments
    *,
    generator: Generator[Any, Any, Any],
    observer: Observer,
    module: str,
    version: str,
    function_name: str,
    called_start: datetime,
    original_args: tuple[Any, ...],
    original_kwargs: dict[str, Any],
    nebuly_kwargs: dict[str, Any],
) -> Generator[Any, Any, Any]:
    """
    Watch a generator

    Creates the SpanWatch object while the generator is being iterated over.
    Waits until the iteration is done to call the _observer.
    """
    original_result = []
    generator_first_element_timestamp = None

    first_element = True
    for element in generator:
        if first_element:
            first_element = False
            generator_first_element_timestamp = datetime.now(timezone.utc)
        logger.info("Yielding %s", element)
        original_result.append(deepcopy(element))
        yield element

    called_end = datetime.now(timezone.utc)

    watched = SpanWatch(
        module=module,
        version=version,
        function=function_name,
        called_start=called_start,
        called_end=called_end,
        called_with_args=original_args,
        called_with_kwargs=original_kwargs,
        returned=original_result,
        generator=True,
        generator_first_element_timestamp=generator_first_element_timestamp,
        provider_extras=nebuly_kwargs,
    )

    _add_interaction_span(
        original_args=original_args,
        original_kwargs=original_kwargs,
        module=module,
        function_name=function_name,
        output=original_result,
        observer=observer,
        watched=watched,
        nebuly_kwargs=nebuly_kwargs,
        stream=True,
    )


async def watch_from_generator_async(  # pylint: disable=too-many-arguments
    *,
    generator: AsyncGenerator[Any, Any],
    observer: Observer,
    module: str,
    version: str,
    function_name: str,
    called_start: datetime,
    original_args: tuple[Any, ...],
    original_kwargs: dict[str, Any],
    nebuly_kwargs: dict[str, Any],
) -> AsyncGenerator[Any, Any]:
    """
    Watch a generator

    Creates the SpanWatch object while the generator is being iterated over.
    Waits until the iteration is done to call the _observer.
    """
    original_result = []
    generator_first_element_timestamp = None

    first_element = True
    async for element in generator:
        if first_element:
            first_element = False
            generator_first_element_timestamp = datetime.now(timezone.utc)
        logger.info("Yielding %s", element)
        original_result.append(deepcopy(element))
        yield element

    called_end = datetime.now(timezone.utc)

    watched = SpanWatch(
        module=module,
        version=version,
        function=function_name,
        called_start=called_start,
        called_end=called_end,
        called_with_args=original_args,
        called_with_kwargs=original_kwargs,
        returned=original_result,
        generator=True,
        generator_first_element_timestamp=generator_first_element_timestamp,
        provider_extras=nebuly_kwargs,
    )
    _add_interaction_span(
        original_args=original_args,
        original_kwargs=original_kwargs,
        module=module,
        function_name=function_name,
        output=original_result,
        observer=observer,
        watched=watched,
        nebuly_kwargs=nebuly_kwargs,
        stream=True,
    )


def _handle_unpickleable_objects() -> None:
    """
    Make unpickleable objects pickleable to avoid errors when
    using the deepcopy function
    """
    try:
        from cohere.client import Client  # pylint: disable=import-outside-toplevel

        def _pickle_cohere_client(c: Client) -> tuple[type[Client], tuple[Any, ...]]:
            return Client, (
                c.api_key,
                c.num_workers,
                c.request_dict,
                True,
                None,
                c.max_retries,
                c.timeout,
                c.api_url,
            )

        copyreg.pickle(Client, _pickle_cohere_client)
    except ImportError:
        pass

    try:
        from anthropic import (  # pylint: disable=import-outside-toplevel
            Anthropic,
            AsyncAnthropic,
        )

        def _pickle_anthropic_client(
            c: Anthropic,
        ) -> tuple[type[Anthropic], tuple[Any, ...], dict[str, Any]]:
            return (
                Anthropic,
                (),
                {
                    "auth_token": c.auth_token,
                    "base_url": c.base_url,
                    "api_key": c.api_key,
                    "timeout": c.timeout,
                    "max_retries": c.max_retries,
                    "default_headers": c.default_headers,
                },
            )

        def _pickle_async_anthropic_client(
            c: AsyncAnthropic,
        ) -> tuple[type[AsyncAnthropic], tuple[Any, ...], dict[str, Any]]:
            return (
                AsyncAnthropic,
                (),
                {
                    "auth_token": c.auth_token,
                    "base_url": c.base_url,
                    "api_key": c.api_key,
                    "timeout": c.timeout,
                    "max_retries": c.max_retries,
                    "default_headers": c.default_headers,
                },
            )

        copyreg.pickle(Anthropic, _pickle_anthropic_client)
        copyreg.pickle(AsyncAnthropic, _pickle_async_anthropic_client)

    except ImportError:
        pass

    try:
        from vertexai.language_models import (  # pylint: disable=import-outside-toplevel
            ChatModel,
            TextGenerationModel,
        )

        def _pickle_text_generation_model(
            c: TextGenerationModel,
        ) -> tuple[type[TextGenerationModel], tuple[Any, ...]]:
            return TextGenerationModel, (c._model_id, c._endpoint_name)

        copyreg.pickle(TextGenerationModel, _pickle_text_generation_model)
        copyreg.pickle(ChatModel, _pickle_text_generation_model)
    except ImportError:
        pass


def _setup_args_kwargs(
    *args: tuple[Any, ...], **kwargs: dict[str, Any]
) -> tuple[tuple[Any, ...], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """
    Split nebuly kwargs from function kwargs

    Returns:
        original_args: deepcopy of args passed to the function
        original_kwargs: deepcopy of kwargs passed to the function that are not
            nebuly kwargs
        function_kwargs: kwargs passed to the function that are not nebuly kwargs
        nebuly_kwargs: kwargs passed to the function that are nebuly kwargs
    """
    nebuly_kwargs, function_kwargs = _split_nebuly_kwargs(kwargs)

    _handle_unpickleable_objects()
    original_args = deepcopy(args)
    nebuly_kwargs = deepcopy(nebuly_kwargs)
    original_kwargs = deepcopy(function_kwargs)

    return original_args, original_kwargs, function_kwargs, nebuly_kwargs


def coroutine_wrapper(
    f: Callable[[Any], Any],
    observer: Observer,
    module: str,
    version: str,
    function_name: str,
) -> Callable[[Any], Any]:
    @wraps(f)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger.debug("Calling %s.%s", module, function_name)

        if module == "langchain":
            from nebuly.langchain import wrap_langchain_async

            return wrap_langchain_async(
                function_name=function_name, f=f, args=args, kwargs=kwargs
            )

        (
            original_args,
            original_kwargs,
            function_kwargs,
            nebuly_kwargs,
        ) = _setup_args_kwargs(*args, **kwargs)

        generator_first_element_timestamp = None

        called_start = datetime.now(timezone.utc)

        if isasyncgenfunction(f):
            result = f(*args, **function_kwargs)
        else:
            result = await f(*args, **function_kwargs)

        if isinstance(result, AsyncGen):
            logger.debug("Result is a generator")
            return watch_from_generator_async(
                generator=result,
                observer=observer,
                module=module,
                version=version,
                function_name=function_name,
                called_start=called_start,
                original_args=original_args,
                original_kwargs=original_kwargs,
                nebuly_kwargs=nebuly_kwargs,
            )

        logger.debug("Result is not a generator")

        original_result = deepcopy(result)
        called_end = datetime.now(timezone.utc)
        watched = SpanWatch(
            module=module,
            version=version,
            function=function_name,
            called_start=called_start,
            called_end=called_end,
            called_with_args=original_args,
            called_with_kwargs=original_kwargs,
            returned=original_result,
            generator=False,
            generator_first_element_timestamp=generator_first_element_timestamp,
            provider_extras=nebuly_kwargs,
        )
        _add_interaction_span(
            original_args=original_args,
            original_kwargs=original_kwargs,
            module=module,
            function_name=function_name,
            output=original_result,
            observer=observer,
            watched=watched,
            nebuly_kwargs=nebuly_kwargs,
        )
        return result

    return wrapper


def function_wrapper(
    f: Callable[[Any], Any],
    observer: Observer,
    module: str,
    version: str,
    function_name: str,
) -> Callable[[Any], Any]:
    @wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger.debug("Calling %s.%s", module, function_name)

        if module == "langchain":
            from nebuly.langchain import wrap_langchain

            return wrap_langchain(
                function_name=function_name, f=f, args=args, kwargs=kwargs
            )
        (
            original_args,
            original_kwargs,
            function_kwargs,
            nebuly_kwargs,
        ) = _setup_args_kwargs(*args, **kwargs)

        generator_first_element_timestamp = None

        called_start = datetime.now(timezone.utc)
        result = f(*args, **function_kwargs)

        if isinstance(result, SyncGen):
            logger.debug("Result is a generator")
            return watch_from_generator(
                generator=result,
                observer=observer,
                module=module,
                version=version,
                function_name=function_name,
                called_start=called_start,
                original_args=original_args,
                original_kwargs=original_kwargs,
                nebuly_kwargs=nebuly_kwargs,
            )

        logger.debug("Result is not a generator")

        original_result = deepcopy(result)
        called_end = datetime.now(timezone.utc)
        watched = SpanWatch(
            module=module,
            version=version,
            function=function_name,
            called_start=called_start,
            called_end=called_end,
            called_with_args=original_args,
            called_with_kwargs=original_kwargs,
            returned=original_result,
            generator=False,
            generator_first_element_timestamp=generator_first_element_timestamp,
            provider_extras=nebuly_kwargs,
        )
        _add_interaction_span(
            original_args=original_args,
            original_kwargs=original_kwargs,
            module=module,
            function_name=function_name,
            output=original_result,
            observer=observer,
            watched=watched,
            nebuly_kwargs=nebuly_kwargs,
        )
        return result

    return wrapper


def _patcher(
    observer: Observer, module: str, version: str, function_name: str
) -> Callable[[Any], Any]:
    """
    Decorator that calls _observer with a SpanWatch instance when the decorated
    function is called

    kwargs that start with nebuly_ are passed to the _observer and not the
    decorated function
    """

    def inner(f: Callable[[Any], Any]) -> Callable[[Any], Any]:
        if iscoroutinefunction(f) or isasyncgenfunction(f):
            return coroutine_wrapper(f, observer, module, version, function_name)

        return function_wrapper(f, observer, module, version, function_name)

    return inner
