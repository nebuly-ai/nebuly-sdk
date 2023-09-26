from nebuly.config import PACKAGES
from nebuly.handlers import set_tracking_handlers
from nebuly.monkey_patching import import_and_patch_packages


def nebuly_init(observer):
    import_and_patch_packages(PACKAGES, observer)
    set_tracking_handlers()
