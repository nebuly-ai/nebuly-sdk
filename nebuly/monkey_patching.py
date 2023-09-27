from __future__ import annotations

import importlib
import logging
import sys
from copy import deepcopy
from datetime import datetime, timezone
from functools import wraps
from inspect import isasyncgenfunction, iscoroutinefunction
from types import ModuleType
from typing import Any, AsyncGenerator, Callable, Generator, Iterable

from nebuly.contextmanager import (
    NotInInteractionContext,
    get_nearest_open_interaction,
    new_interaction,
)
from nebuly.entities import Observer, Package, SpanWatch

logger = logging.getLogger(__name__)


def check_no_packages_already_imported(packages: Iterable[Package]) -> None:
    """
    Check that no packages in packages have already been imported.
    """
    for package in packages:
        if package.name in sys.modules:
            logger.warning("%s already imported", package.name)


def import_and_patch_packages(packages: Iterable[Package], observer: Observer) -> None:
    """
    Import each package in packages and patch it with the observer.
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
        from nebuly.providers.openai import extract_openai_output

        return extract_openai_output(function_name, output)
    if module == "cohere":
        from nebuly.providers.cohere import extract_cohere_output

        return extract_cohere_output(function_name, output)
    if module == "anthropic":
        from nebuly.providers.anthropic import extract_anthropic_output

        return extract_anthropic_output(function_name, output)
    if module == "huggingface_hub":
        from nebuly.providers.huggingface_hub import extract_hf_hub_output

        return extract_hf_hub_output(function_name, output)
    if module == "google":
        from nebuly.providers.google import extract_google_output

        return extract_google_output(function_name, output)
    if module == "vertexai":
        from nebuly.providers.vertexai import extract_vertexai_output

        return extract_vertexai_output(function_name, output)
    return str(output)


def _extract_output_generator(outputs: Any, module: str, function_name: str) -> str:
    if module == "openai":
        from nebuly.providers.openai import extract_openai_output_generator

        return extract_openai_output_generator(function_name, outputs)
    if module == "cohere":
        from nebuly.providers.cohere import extract_cohere_output_generator

        return extract_cohere_output_generator(function_name, outputs)
    if module == "anthropic":
        from nebuly.providers.anthropic import extract_anthropic_output_generator

        return extract_anthropic_output_generator(function_name, outputs)
    if module == "vertexai":
        from nebuly.providers.vertexai import extract_vertexai_output_generator

        return extract_vertexai_output_generator(function_name, outputs)


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


def _extract_input_and_history(
    original_args: tuple[Any, ...],
    original_kwargs: dict[str, Any],
    module: str,
    function_name: str,
) -> tuple[str, list[tuple[str, Any]]]:
    if module == "openai":
        from nebuly.providers.openai import extract_openai_input_and_history

        return extract_openai_input_and_history(original_kwargs, function_name)
    if module == "cohere":
        from nebuly.providers.cohere import extract_cohere_input_and_history

        return extract_cohere_input_and_history(
            original_args, original_kwargs, function_name
        )
    if module == "anthropic":
        from nebuly.providers.anthropic import extract_anthropic_input_and_history

        return extract_anthropic_input_and_history(original_kwargs, function_name)
    if module == "huggingface_hub":
        from nebuly.providers.huggingface_hub import extract_hf_hub_input_and_history

        return extract_hf_hub_input_and_history(
            original_args, original_kwargs, function_name
        )
    if module == "google":
        from nebuly.providers.google import extract_google_input_and_history

        return extract_google_input_and_history(
            original_args, original_kwargs, function_name
        )
    if module == "vertexai":
        from nebuly.providers.vertexai import extract_vertexai_input_and_history

        return extract_vertexai_input_and_history(
            original_args, original_kwargs, function_name
        )


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
    Waits until the iteration is done to call the observer.
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
    Waits until the iteration is done to call the observer.
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
        from nebuly.providers.cohere import handle_cohere_unpickable_objects

        handle_cohere_unpickable_objects()
    except ImportError:
        pass

    try:
        from nebuly.providers.anthropic import handle_anthropic_unpickable_objects

        handle_anthropic_unpickable_objects()
    except ImportError:
        pass

    try:
        from nebuly.providers.vertexai import handle_vertexai_unpickable_objects

        handle_vertexai_unpickable_objects()
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


def _is_generator(obj: Any):
    if isinstance(obj, (Generator, AsyncGenerator)):
        return True

    try:
        from nebuly.providers.cohere import is_cohere_generator

        if is_cohere_generator(obj):
            return True
    except ImportError:
        pass

    try:
        from nebuly.providers.anthropic import is_anthropic_generator

        if is_anthropic_generator(obj):
            return True
    except ImportError:
        pass


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
            from nebuly.providers.langchain import wrap_langchain_async

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

        if _is_generator(result):
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
            from nebuly.providers.langchain import wrap_langchain

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

        if _is_generator(result):
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
    Decorator that calls observer with a SpanWatch instance when the decorated
    function is called

    kwargs that start with nebuly_ are passed to the observer and not the
    decorated function
    """

    def inner(f: Callable[[Any], Any]) -> Callable[[Any], Any]:
        if iscoroutinefunction(f) or isasyncgenfunction(f):
            return coroutine_wrapper(f, observer, module, version, function_name)

        return function_wrapper(f, observer, module, version, function_name)

    return inner
