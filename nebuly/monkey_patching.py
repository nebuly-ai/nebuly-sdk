from __future__ import annotations

import importlib
import logging
import sys
from copy import deepcopy
from datetime import datetime, timezone
from functools import wraps
from inspect import isasyncgenfunction, iscoroutinefunction
from types import ModuleType
from typing import Any, AsyncGenerator, Callable, Generator, Iterable, cast
from uuid import UUID

from langchain.callbacks.manager import CallbackManager

from nebuly.entities import Observer, Package, Watched
from nebuly.handlers import LangChainTrackingHandler

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
        except AttributeError:
            logger.warning("Failed to patch %s", attr)


def _monkey_patch_attribute(attr: str, module: ModuleType, observer: Observer) -> None:
    version = module.__version__ if hasattr(module, "__version__") else "unknown"
    tmp_component = module
    path = attr.split(".")
    for component_name in path[:-1]:
        tmp_component = getattr(tmp_component, component_name)

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

    Creates the Watched object while the generator is being iterated over.
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

    watched = Watched(
        module=module,
        version=version,
        function=function_name,
        called_start=called_start,
        called_end=called_end,
        called_with_args=original_args,
        called_with_kwargs=original_kwargs,
        called_with_nebuly_kwargs=nebuly_kwargs,
        returned=original_result,
        generator=True,
        generator_first_element_timestamp=generator_first_element_timestamp,
    )
    observer(watched)


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

    Creates the Watched object while the generator is being iterated over.
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

    watched = Watched(
        module=module,
        version=version,
        function=function_name,
        called_start=called_start,
        called_end=called_end,
        called_with_args=original_args,
        called_with_kwargs=original_kwargs,
        called_with_nebuly_kwargs=nebuly_kwargs,
        returned=original_result,
        generator=True,
        generator_first_element_timestamp=generator_first_element_timestamp,
    )
    observer(watched)


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

    original_args = deepcopy(args)
    nebuly_kwargs = deepcopy(nebuly_kwargs)
    original_kwargs = deepcopy(function_kwargs)

    return original_args, original_kwargs, function_kwargs, nebuly_kwargs


def _add_tracking_info_to_provider_call(
    f: Callable[[Any], Any], *args: Any, **kwargs: Any
) -> Callable[[Any], Any]:
    """
    This function is a hack to add the chain run_ids info to the kwargs of the
    provider call. This is needed to associate each call to a provider (ex OpenAI)
    with the belonging chain.
    """
    callbacks = kwargs.get("callbacks")
    if not isinstance(callbacks, CallbackManager) or callbacks.parent_run_id is None:
        # If the llm/chat_model is called without a CallbackManager, it's not
        # called from a chain, so we don't need to add the run ids info
        return f(*args, **kwargs)  # type: ignore

    # Get the parent_run_id from the CallbackManager
    callback_manager = callbacks
    parent_run_id = cast(UUID, callback_manager.parent_run_id)
    additional_kwargs = {"nebuly_parent_run_id": parent_run_id}

    # Get the root_run_id from the LangChainTrackingHandler
    for handler in callback_manager.handlers:
        if isinstance(handler, LangChainTrackingHandler):
            event_pairing_dispatcher = handler.event_pairing_dispatcher
            root_run_id = event_pairing_dispatcher.events_storage.get_root_id(
                parent_run_id
            )
            additional_kwargs["nebuly_root_run_id"] = root_run_id
            break

    return f(*args, **kwargs, **additional_kwargs)  # type: ignore


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

        if module == "langchain" and function_name.startswith(
            ("llms.base.BaseLLM", "chat_models.base.BaseChatModel")
        ):
            func = _add_tracking_info_to_provider_call(f, *args, **kwargs)
            if isasyncgenfunction(func):
                return func(*args, **kwargs)
            return await func(*args, **kwargs)

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

        if isinstance(result, AsyncGenerator):
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
        watched = Watched(
            module=module,
            version=version,
            function=function_name,
            called_start=called_start,
            called_end=called_end,
            called_with_args=original_args,
            called_with_kwargs=original_kwargs,
            called_with_nebuly_kwargs=nebuly_kwargs,
            returned=original_result,
            generator=False,
            generator_first_element_timestamp=generator_first_element_timestamp,
        )
        observer(watched)
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

        if module == "langchain" and function_name.startswith(
            ("llms.base.BaseLLM", "chat_models.base.BaseChatModel")
        ):
            return _add_tracking_info_to_provider_call(f, *args, **kwargs)
        (
            original_args,
            original_kwargs,
            function_kwargs,
            nebuly_kwargs,
        ) = _setup_args_kwargs(*args, **kwargs)

        generator_first_element_timestamp = None

        called_start = datetime.now(timezone.utc)
        result = f(*args, **function_kwargs)

        if isinstance(result, Generator):
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
        watched = Watched(
            module=module,
            version=version,
            function=function_name,
            called_start=called_start,
            called_end=called_end,
            called_with_args=original_args,
            called_with_kwargs=original_kwargs,
            called_with_nebuly_kwargs=nebuly_kwargs,
            returned=original_result,
            generator=False,
            generator_first_element_timestamp=generator_first_element_timestamp,
        )
        observer(watched)
        return result

    return wrapper


def _patcher(
    observer: Observer, module: str, version: str, function_name: str
) -> Callable[[Any], Any]:
    """
    Decorator that calls observer with a Watched instance when the decorated
    function is called

    kwargs that start with nebuly_ are passed to the observer and not the
    decorated function
    """

    def inner(f: Callable[[Any], Any]) -> Callable[[Any], Any]:
        if iscoroutinefunction(f) or isasyncgenfunction(f):
            return coroutine_wrapper(f, observer, module, version, function_name)

        return function_wrapper(f, observer, module, version, function_name)

    return inner
