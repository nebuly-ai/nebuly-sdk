from nebuly.entities import Package
from nebuly.monkey_patcher import (check_no_packages_already_imported,
                                   import_and_patch_packages)
from nebuly.observer import NebulyObserver

PACKAGES = [Package("openai", ["0.10.2"], ["Completion.create"])]


def init(*, api_key: str, project: str, phase: str) -> None:
    check_no_packages_already_imported(PACKAGES)
    observer = NebulyObserver(
        api_key=api_key,
        project=project,
        phase=phase,
        publisher=lambda _: None,
    )
    import_and_patch_packages(PACKAGES, observer.observe)
