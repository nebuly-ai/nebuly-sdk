import atexit
import contextlib
import copy
from typing import Generator, Optional, Any, List

from nebuly.core.schemas import Task, DevelopmentPhase, TagData
from nebuly.core.clients import NebulyTrackingDataThread, NebulyClient
from nebuly.core.queues import NebulyQueue, Tracker
from nebuly.utils.logger import nebuly_logger

_nebuly_queue: Optional[NebulyQueue] = None


def init(
    project: str,
    phase: DevelopmentPhase,
    task: Task = Task.UNKNOWN,
) -> None:
    """Initialize Nebuly SDK, start tracking data and replace SDK functions.
    Call this method before your main() function.

    Args:
        project (str): Name of the project
        phase (DevelopmentPhase): Development phase of the project
        task (Optional[Enum], optional): Task being performed.
            Defaults to Task.UNDETECTED.

    Raises:
        TypeError: If project is not of type str
        TypeError: If phase is not of type DevelopmentPhase
        TypeError: If task is not of type Optional[Task]
    """

    _check_input_types(project=project, phase=phase, task=task)
    from nebuly import api_key

    if api_key is None:
        nebuly_logger.warning(
            msg=(
                "Nebuly API key not found. "
                "Please set NEBULY_API_KEY environment variable."
            )
        )
        return

    tag_data = TagData(
        project=project,
        phase=phase,
        task=task,
    )
    global _nebuly_queue
    _nebuly_queue = NebulyQueue(tag_data=tag_data)

    nebuly_tracking_thread = NebulyTrackingDataThread(
        queue=_nebuly_queue, nebuly_client=NebulyClient(api_key=api_key)
    )
    nebuly_tracking_thread.daemon = True
    atexit.register(_stop_thread_when_main_ends, nebuly_tracking_thread)
    nebuly_tracking_thread.start()

    tracker_list: list[Tracker] = _instantiate_trackers(nebuly_queue=_nebuly_queue)
    for tracker in tracker_list:
        tracker.replace_sdk_functions()


@contextlib.contextmanager
def tracker(
    project: str = "unknown",
    phase: DevelopmentPhase = DevelopmentPhase.UNKNOWN,
    task: Task = Task.UNKNOWN,
) -> Generator[None, Any, None]:
    """Context manager to temporarily replace the tracker info.
    This is useful when you want to track data for a different project, phase or task.
    within the same main() function.

    Args:
        project (Optional[str], optional): Name of the project.
            Defaults to None.
        phase (Optional[DevelopmentPhase], optional): Development phase of the project.
            Defaults to None.
        task (Optional[Task], optional): Task being performed.

    Raises:
        TypeError: If project is not of type Optional[str]
        TypeError: If phase is not of type Optional[DevelopmentPhase]
        TypeError: If task is not of type Optional[Task]
    """
    _check_input_types(project=project, phase=phase, task=task)

    if _nebuly_queue is None:
        raise RuntimeError("Please call nebuly.init() before using nebuly.tracker()")

    old_tag_data: TagData = copy.deepcopy(x=_nebuly_queue.tag_data)
    new_tag_data = TagData(
        project=project,
        phase=phase,
        task=task,
    )
    _nebuly_queue.patch_tag_data(tag_data=new_tag_data)
    yield
    _nebuly_queue.patch_tag_data(tag_data=old_tag_data)


def _instantiate_trackers(nebuly_queue: NebulyQueue) -> List[Tracker]:
    tracker_list: List[Tracker] = []
    try:
        from nebuly.trackers.openai import OpenAITracker

        tracker_list.append(OpenAITracker(nebuly_queue=nebuly_queue))
    except ImportError:
        pass

    return tracker_list


def _check_input_types(
    project: str, phase: DevelopmentPhase, task: Optional[Task]
) -> None:
    if isinstance(project, str) is False:  # type: ignore
        raise TypeError(f"project must be of type str, not {type(project)}")
    if isinstance(phase, DevelopmentPhase) is False:  # type: ignore
        raise TypeError(f"phase must be of type DevelopmentPhase, not {type(phase)}")
    if isinstance(task, Task) is False:  # type: ignore
        raise TypeError(f"task must be of type Task, not {type([task])}")


def _stop_thread_when_main_ends(
    nebuly_thread_instance: NebulyTrackingDataThread,
    stop_attempts: int = 0,
) -> None:
    try:
        nebuly_thread_instance.thread_running = False
        nebuly_thread_instance.join()
    except KeyboardInterrupt:
        if stop_attempts < 2:
            stop_attempts += 1
            _stop_thread_when_main_ends(
                nebuly_thread_instance=nebuly_thread_instance,
                stop_attempts=stop_attempts,
            )
        else:
            nebuly_thread_instance.force_exit = True
            nebuly_thread_instance.join()
