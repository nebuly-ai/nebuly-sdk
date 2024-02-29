from __future__ import annotations

import asyncio
import importlib
import json
import logging
import sys
from copy import deepcopy
from datetime import datetime, timezone
from functools import wraps
from importlib.metadata import version
from inspect import isasyncgenfunction, iscoroutinefunction
from types import ModuleType
from typing import Any, AsyncGenerator, Callable, Generator, Iterable, cast

from packaging.version import parse as parse_version

from nebuly.config import NEBULY_KWARGS
from nebuly.contextmanager import (
    InteractionContext,
    get_nearest_open_interaction,
    new_interaction,
)
from nebuly.entities import HistoryEntry, ModelInput, Observer, Package, SpanWatch
from nebuly.exceptions import NebulyException, NotInInteractionContext
from nebuly.providers.base import ProviderDataExtractor

logger = logging.getLogger(__name__)


def check_no_packages_already_imported(packages: Iterable[Package]) -> None:
    """
    Check that no packages in packages have already been imported.
    """
    for package in packages:
        if package.name in sys.modules:
            logger.debug("%s already imported", package.name)


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
    try:
        package_version = version(package.name)
    except Exception:  # pylint: disable=broad-except
        package_version = "unknown"

    if package_version != "unknown":
        max_version_str = package.versions.max_version
        min_version = parse_version(package.versions.min_version)
        pkg_version = parse_version(package_version)

        if max_version_str is None and (
            min_version
            > pkg_version
            != min_version.base_version
            != pkg_version.base_version
        ):
            return
        if (
            max_version_str is not None
            and not min_version <= pkg_version < parse_version(max_version_str)
        ):
            return

    for attr in package.to_patch:
        try:
            _monkey_patch_attribute(attr, module, package_version, observer)
        except (AttributeError, ImportError):
            logger.debug("Failed to patch %s", attr)


def _monkey_patch_attribute(
    attr: str, module: ModuleType, package_version: str, observer: Observer
) -> None:
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
        _patcher(observer, module.__name__, package_version, attr)(
            getattr(tmp_component, path[-1])
        ),
    )


def _extract_nebuly_kwargs_from_arg(arg: Any, nebuly_kwargs: dict[str, Any]) -> Any:
    if not isinstance(arg, (dict, str)):
        return arg

    needs_json_conversion = False
    if isinstance(arg, str):
        try:
            arg = json.loads(arg)
            if not isinstance(arg, dict):
                # Need this to handle the case when the arg is a string containing
                # for example an int
                return json.dumps(arg)

            needs_json_conversion = True
        except json.JSONDecodeError:
            return arg

    for key in NEBULY_KWARGS:
        if key in arg:
            nebuly_kwargs[key] = arg.pop(key)

    for key, value in arg.items():
        arg[key] = _extract_nebuly_kwargs_from_arg(value, nebuly_kwargs)

    if needs_json_conversion:
        arg = json.dumps(arg)

    return arg


def _split_nebuly_kwargs(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Split nebuly kwargs from function kwargs
    """

    nebuly_kwargs = {}
    function_kwargs = {}
    if len(kwargs) > 0:
        for key in kwargs:
            if key in NEBULY_KWARGS:
                nebuly_kwargs[key] = kwargs[key]
            else:
                function_kwargs[key] = kwargs[key]
    else:
        for arg in args:
            arg = _extract_nebuly_kwargs_from_arg(arg, nebuly_kwargs)
    return nebuly_kwargs, function_kwargs


def _get_provider_data_extractor(
    module: str,
    original_args: tuple[Any, ...],
    original_kwargs: dict[str, Any],
    function_name: str,
) -> ProviderDataExtractor:
    constructor: type[ProviderDataExtractor] | None = None
    if module == "openai":
        if (
            parse_version(version(module)) < parse_version("1.0.0")
            and parse_version(version(module)).base_version != "1.0.0"
        ):
            from nebuly.providers.openai_legacy import (  # pylint: disable=import-outside-toplevel  # noqa: E501
                OpenAILegacyDataExtractor,
            )

            constructor = OpenAILegacyDataExtractor
        else:
            from nebuly.providers.openai import (  # pylint: disable=import-outside-toplevel  # noqa: E501
                OpenAIDataExtractor,
            )

            constructor = OpenAIDataExtractor
    if module == "cohere":
        from nebuly.providers.cohere import (  # pylint: disable=import-outside-toplevel
            CohereDataExtractor,
        )

        constructor = CohereDataExtractor
    if module == "anthropic":
        from nebuly.providers.anthropic import (  # pylint: disable=import-outside-toplevel  # noqa: E501
            AnthropicDataExtractor,
        )

        constructor = AnthropicDataExtractor

    if module == "google":
        from nebuly.providers.google import (  # pylint: disable=import-outside-toplevel  # noqa: E501
            GoogleDataExtractor,
        )

        constructor = GoogleDataExtractor

    if module == "transformers":
        from nebuly.providers.huggingface import (  # pylint: disable=import-outside-toplevel  # noqa: E501
            HuggingFaceDataExtractor,
        )

        constructor = HuggingFaceDataExtractor

    if module == "huggingface_hub":
        from nebuly.providers.huggingface_hub import (  # pylint: disable=import-outside-toplevel  # noqa: E501
            HuggingFaceHubDataExtractor,
        )

        constructor = HuggingFaceHubDataExtractor

    if module == "vertexai":
        from nebuly.providers.vertexai import (  # pylint: disable=import-outside-toplevel  # noqa: E501
            VertexAIDataExtractor,
        )

        constructor = VertexAIDataExtractor

    if module == "botocore":
        from nebuly.providers.aws_bedrock import (  # pylint: disable=import-outside-toplevel  # noqa: E501
            AWSBedrockDataExtractor,
        )

        constructor = AWSBedrockDataExtractor

    if constructor is not None:
        return constructor(
            original_args=original_args,
            original_kwargs=original_kwargs,
            function_name=function_name,
        )

    raise ValueError(f"Unknown module: {module}")


def _add_span_to_interaction(  # pylint: disable=too-many-arguments
    observer: Observer,
    interaction: InteractionContext,
    user_input: str,
    history: list[HistoryEntry],
    output: str,
    watched: SpanWatch,
    nebuly_kwargs: dict[str, Any],
) -> None:
    interaction._set_observer(observer)  # pylint: disable=protected-access
    interaction._add_span(watched)  # pylint: disable=protected-access
    if interaction.input is None:
        interaction.set_input(user_input)
    if interaction.history is None:
        interaction.set_history(history)
    if interaction.output is None:
        interaction.set_output(output)
    user: str | None = nebuly_kwargs.get("user_id")
    if interaction.user is None and user is not None:
        interaction._set_user(user)  # pylint: disable=protected-access
    user_group_profile: str | None = nebuly_kwargs.get("user_group_profile")
    if interaction.user_group_profile is None and user_group_profile is not None:
        interaction._set_user_group_profile(  # pylint: disable=protected-access
            user_group_profile
        )
    if "nebuly_tags" in nebuly_kwargs:
        interaction._add_tags(  # pylint: disable=protected-access
            nebuly_kwargs["nebuly_tags"]
        )
    if "feature_flags" in nebuly_kwargs:
        interaction._add_feature_flags(  # pylint: disable=protected-access
            nebuly_kwargs["feature_flags"]
        )
    feature_flags: list[str] | None = nebuly_kwargs.get("feature_flag")
    if interaction.feature_flags is None and feature_flags is not None:
        interaction.feature_flags = feature_flags
    if "nebuly_api_key" in nebuly_kwargs:
        interaction._set_api_key(  # pylint: disable=protected-access
            nebuly_kwargs["nebuly_api_key"]
        )


def _add_interaction_span(  # pylint: disable=too-many-arguments, too-many-locals
    original_args: tuple[Any, ...],
    original_kwargs: dict[str, Any],
    module: str,
    function_name: str,
    output: Any,
    observer: Observer,
    watched: SpanWatch,
    nebuly_kwargs: dict[str, Any],
    start_time: datetime,
    stream: bool = False,
) -> None:
    try:
        provider_data_extractor = _get_provider_data_extractor(
            module, original_args, original_kwargs, function_name
        )
        model_input_res = provider_data_extractor.extract_input_and_history(output)
        model_output_res = provider_data_extractor.extract_output(stream, output)
        media = provider_data_extractor.extract_media()

        # Add media to the watched object
        if media is not None:
            watched.media = media
    # FIXME: this is ignoring the model_input_res when the model_output_res throws an
    # exception.
    except ValueError:
        logger.debug("Unknown module: %s", function_name)
        model_input_res = ModelInput(prompt="")
        model_output_res = ""

    try:
        if "nebuly_interaction" in nebuly_kwargs:
            interaction = nebuly_kwargs["nebuly_interaction"]
        else:
            interaction = get_nearest_open_interaction()
        if isinstance(model_input_res, list):
            logger.warning(
                "Batch prediction found inside a user defined interaction, only the "
                "first element will be used"
            )
            model_input_res = model_input_res[0]
            model_output_res = model_output_res[0]

        _add_span_to_interaction(
            observer=observer,
            interaction=interaction,
            user_input=model_input_res.prompt,
            history=model_input_res.history,
            output=cast(str, model_output_res),
            watched=watched,
            nebuly_kwargs=nebuly_kwargs,
        )
    except NotInInteractionContext:
        if isinstance(model_input_res, list):
            for model_input, model_output in zip(model_input_res, model_output_res):
                with new_interaction() as interaction:
                    interaction.time_start = start_time
                    _add_span_to_interaction(
                        observer=observer,
                        interaction=interaction,
                        user_input=model_input.prompt,
                        history=model_input.history,
                        output=model_output,
                        watched=watched,
                        nebuly_kwargs=nebuly_kwargs,
                    )
        else:
            with new_interaction() as interaction:
                interaction.time_start = start_time
                _add_span_to_interaction(
                    observer=observer,
                    interaction=interaction,
                    user_input=model_input_res.prompt,
                    history=model_input_res.history,
                    output=cast(str, model_output_res),
                    watched=watched,
                    nebuly_kwargs=nebuly_kwargs,
                )


def watch_from_generator(  # pylint: disable=too-many-arguments
    *,
    generator: Generator[Any, Any, Any],
    observer: Observer,
    module: str,
    package_version: str,
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
        logger.debug("Yielding %s", element)
        original_result.append(deepcopy(element))
        yield element

    called_end = datetime.now(timezone.utc)

    watched = SpanWatch(
        module=module,
        version=package_version,
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
        start_time=called_start,
    )


async def watch_from_generator_async(  # pylint: disable=too-many-arguments
    *,
    generator: AsyncGenerator[Any, Any],
    observer: Observer,
    module: str,
    package_version: str,
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
        logger.debug("Yielding %s", element)
        original_result.append(deepcopy(element))
        yield element

    called_end = datetime.now(timezone.utc)

    watched = SpanWatch(
        module=module,
        version=package_version,
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
        start_time=called_start,
    )


def _handle_unpickleable_objects(module: str, args: Any) -> None:
    """
    Make unpickleable objects pickleable to avoid errors when
    using the deepcopy function
    """
    try:
        from nebuly.providers.openai import (  # pylint: disable=import-outside-toplevel
            handle_openai_unpickable_objects,
        )

        handle_openai_unpickable_objects()
    except ImportError:
        pass
    try:
        from nebuly.providers.cohere import (  # pylint: disable=import-outside-toplevel
            handle_cohere_unpickable_objects,
        )

        handle_cohere_unpickable_objects()
    except ImportError:
        pass

    try:
        from nebuly.providers.anthropic import (  # pylint: disable=import-outside-toplevel  # noqa: E501
            handle_anthropic_unpickable_objects,
        )

        handle_anthropic_unpickable_objects()
    except ImportError:
        pass

    try:
        from nebuly.providers.google import (  # pylint: disable=import-outside-toplevel
            handle_google_unpickable_objects,
        )

        handle_google_unpickable_objects()
    except ImportError:
        pass

    try:
        from nebuly.providers.vertexai import (  # pylint: disable=import-outside-toplevel  # noqa: E501
            handle_vertexai_unpickable_objects,
        )

        handle_vertexai_unpickable_objects()
    except ImportError:
        pass
    if module == "botocore":
        try:
            from nebuly.providers.aws_bedrock import (  # pylint: disable=import-outside-toplevel  # noqa: E501
                handle_aws_bedrock_unpickable_objects,
            )

            handle_aws_bedrock_unpickable_objects(args[0])
        except ImportError:
            pass


def _setup_args_kwargs(
    *args: Any, **kwargs: Any
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
    nebuly_kwargs, function_kwargs = _split_nebuly_kwargs(args, kwargs)
    original_args = deepcopy(args)
    original_kwargs = deepcopy(function_kwargs)

    return original_args, original_kwargs, function_kwargs, nebuly_kwargs


def _is_generator(obj: Any, module: str) -> bool:
    if isinstance(obj, (Generator, AsyncGenerator)):
        return True

    try:
        from nebuly.providers.openai import (  # pylint: disable=import-outside-toplevel
            is_openai_generator,
        )

        if is_openai_generator(obj):
            return True
    except ImportError:
        pass

    try:
        from nebuly.providers.cohere import (  # pylint: disable=import-outside-toplevel
            is_cohere_generator,
        )

        if is_cohere_generator(obj):
            return True
    except ImportError:
        pass

    try:
        from nebuly.providers.anthropic import (  # pylint: disable=import-outside-toplevel  # noqa: E501
            is_anthropic_generator,
        )

        if is_anthropic_generator(obj):
            return True
    except ImportError:
        pass

    if module == "botocore":
        try:
            from nebuly.providers.aws_bedrock import (  # pylint: disable=import-outside-toplevel  # noqa: E501
                is_aws_bedrock_generator,
            )

            if is_aws_bedrock_generator(obj):
                return True
        except ImportError:
            pass

    return False


def _is_in_context_manager() -> bool:
    try:
        get_nearest_open_interaction()
        return True
    except NotInInteractionContext:
        return False


def coroutine_wrapper(
    f: Callable[[Any], Any],
    observer: Observer,
    module: str,
    package_version: str,
    function_name: str,
) -> Callable[[Any], Any]:
    @wraps(f)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = None
        try:
            logger.debug("Calling %s.%s", module, function_name)

            _handle_unpickleable_objects(module, args)

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
            if len(nebuly_kwargs) == 0 and not _is_in_context_manager():
                return result

            if _is_generator(result, module):
                logger.debug("Result is a generator")
                return watch_from_generator_async(
                    generator=result,
                    observer=observer,
                    module=module,
                    package_version=package_version,
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
                version=package_version,
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
                start_time=called_start,
            )
            return result
        except NebulyException as e:
            raise e
        except Exception as e:  # pylint: disable=broad-except
            logger.error("An error occurred when tracking the function: %s", e)
            if result is not None:
                # Original call was successful, just return the result
                return result
            # call the original function
            _, function_kwargs = _split_nebuly_kwargs(args, kwargs)
            if isasyncgenfunction(f):
                return f(*args, **function_kwargs)
            return await f(*args, **function_kwargs)

    return wrapper


def function_wrapper(
    f: Callable[[Any], Any],
    observer: Observer,
    module: str,
    package_version: str,
    function_name: str,
) -> Callable[[Any], Any]:
    @wraps(f)
    # pylint: disable=too-many-return-statements
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = None
        try:
            logger.debug("Calling %s.%s", module, function_name)

            if module == "botocore":
                from nebuly.providers.aws_bedrock import (  # pylint: disable=import-outside-toplevel  # noqa: E501
                    is_model_supported,
                )

                if args[1] not in [
                    "InvokeModel",
                    "InvokeModelWithResponseStream",
                ] or not is_model_supported(args[2]["modelId"]):
                    for key in NEBULY_KWARGS:
                        if key in args[2]:
                            args[2].pop(key)
                    return f(*args, **kwargs)

            # FIXME: This is being performed for every function call, it should be
            # performed only once
            _handle_unpickleable_objects(module, args)

            (
                original_args,
                original_kwargs,
                function_kwargs,
                nebuly_kwargs,
            ) = _setup_args_kwargs(*args, **kwargs)

            generator_first_element_timestamp = None

            called_start = datetime.now(timezone.utc)
            result = f(*args, **function_kwargs)
            if len(nebuly_kwargs) == 0 and not _is_in_context_manager():
                return result

            if _is_generator(result, module):
                logger.debug("Result is a generator")
                if module == "botocore":
                    # AWS case must be handled separatly because has a
                    # different return type
                    result["body"] = watch_from_generator(
                        generator=result["body"],
                        observer=observer,
                        module=module,
                        package_version=package_version,
                        function_name=function_name,
                        called_start=called_start,
                        original_args=original_args,
                        original_kwargs=original_kwargs,
                        nebuly_kwargs=nebuly_kwargs,
                    )
                    return result
                return watch_from_generator(
                    generator=result,
                    observer=observer,
                    module=module,
                    package_version=package_version,
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
                version=package_version,
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
                start_time=called_start,
            )
            return result
        except NebulyException as e:
            raise e
        except Exception as e:  # pylint: disable=broad-except
            logger.error("An error occurred when tracking the function: %s", e)
            if result is not None:
                # Original call was successful, just return the result
                return result
            # call the original function
            _, function_kwargs = _split_nebuly_kwargs(args, kwargs)
            return f(*args, **function_kwargs)

    return wrapper


def _patcher(
    observer: Observer, module: str, package_version: str, function_name: str
) -> Callable[[Any], Any]:
    """
    Decorator that calls observer with a SpanWatch instance when the decorated
    function is called

    kwargs that start with nebuly_ are passed to the observer and not the
    decorated function
    """

    def inner(f: Callable[[Any], Any]) -> Callable[[Any], Any]:
        if (
            iscoroutinefunction(f)
            or asyncio.iscoroutinefunction(f)  # Needed for python 3.9
            or isasyncgenfunction(f)
            or "async" in function_name.lower()
        ):
            return coroutine_wrapper(
                f, observer, module, package_version, function_name
            )

        return function_wrapper(f, observer, module, package_version, function_name)

    return inner
