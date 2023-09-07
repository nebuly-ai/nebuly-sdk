import importlib
import logging
import sys
from copy import deepcopy
from datetime import datetime, timezone
from functools import wraps
from types import ModuleType
from typing import Any, Callable, Generator

from nebuly.entities import Observer_T, Package, Watched
from nebuly.exceptions import AlreadyImportedError

logger = logging.getLogger(__name__)


def check_no_packages_already_imported(packages: list[Package]) -> None:
    """
    Check that no packages in packages have already been imported.
    """
    for package in packages:
        if package.name in sys.modules:
            raise AlreadyImportedError(
                f"{package.name} already imported. Make sure you initialize",
                "Nebuly before importing any packages.",
            )


def import_and_patch_packages(packages: list[Package], observer: Observer_T) -> None:
    """
    Import each package in packages and patch it with the observer.
    """
    for package in packages:
        try:
            _monkey_patch(package, observer)
        except ImportError:
            pass


def _monkey_patch(package: Package, observer: Observer_T) -> None:
    module = importlib.import_module(package.name)
    for attr in package.to_patch:
        try:
            _monkey_patch_attribute(attr, module, observer)
        except AttributeError:
            logger.warning("Failed to patch %s", attr)


def _monkey_patch_attribute(
    attr: str, module: ModuleType, observer: Observer_T
) -> None:
    tmp_component = module
    path = attr.split(".")
    for component_name in path[:-1]:
        tmp_component = getattr(tmp_component, component_name)

    setattr(
        tmp_component,
        path[-1],
        _patcher(observer, str(module), attr)(getattr(tmp_component, path[-1])),
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
    generator: Generator,
    observer: Observer_T,
    module: str,
    function_name: str,
    called_start: datetime,
    original_args: tuple[Any, ...],
    original_kwargs: dict[str, Any],
    nebuly_kwargs: dict[str, Any],
) -> Generator:
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
        original_result.append(deepcopy(element))
        yield element

    called_end = datetime.now(timezone.utc)

    watched = Watched(
        module=module,
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


def _patcher(observer: Observer_T, module: str, function_name: str) -> Callable:
    """
    Decorator that calls observer with a Watched instance when the decorated
    function is called

    kwargs that start with nebuly_ are passed to the observer and not the
    decorated function
    """

    def inner(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            nebuly_kwargs, function_kwargs = _split_nebuly_kwargs(kwargs)

            original_args = deepcopy(args)
            nebuly_kwargs = deepcopy(nebuly_kwargs)
            original_kwargs = deepcopy(function_kwargs)
            generator_first_element_timestamp = None

            called_start = datetime.now(timezone.utc)
            result = f(*args, **function_kwargs)

            if isinstance(result, Generator):
                return watch_from_generator(
                    result,
                    observer,
                    module,
                    function_name,
                    called_start,
                    original_args,
                    nebuly_kwargs,
                    original_kwargs,
                )

            original_result = deepcopy(result)
            called_end = datetime.now(timezone.utc)
            watched = Watched(
                module=module,
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

    return inner
