from __future__ import annotations

from nebuly.config import PACKAGES
from nebuly.entities import Observer
from nebuly.monkey_patching import import_and_patch_packages
from nebuly.tracking_handlers import set_tracking_handlers


def nebuly_init(observer: Observer) -> None:
    import_and_patch_packages(PACKAGES, observer)
    set_tracking_handlers()