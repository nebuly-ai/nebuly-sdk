from __future__ import annotations

from nebuly.config import PACKAGES
from nebuly.entities import Observer
from nebuly.monkey_patching import import_and_patch_packages


def nebuly_init(observer: Observer) -> None:
    import_and_patch_packages(PACKAGES, observer)
