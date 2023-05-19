import atexit
import contextlib
import copy
from typing import Optional

from nebuly.core.schemas import Task, DevelopmentPhase, TagData
from nebuly.core.clients import NebulyTrackingDataThread, NebulyClient
from nebuly.core.queues import NebulyQueue

_nebuly_queue: Optional[NebulyQueue] = None
api_key = None


def init(
    project: str,
    phase: DevelopmentPhase,
    task: Optional[Task] = Task.UNDETECTED,
):
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

    _check_init_input_types(project, phase, task)

    global _nebuly_queue

    from nebuly import api_key

    _nebuly_queue = NebulyQueue()
    tag_data = TagData(
        project=project,
        phase=phase,
        task=task,
    )
    _nebuly_queue.patch_tag_data(tag_data)

    nebuly_tracking_thread = NebulyTrackingDataThread(
        queue=_nebuly_queue, nebuly_client=NebulyClient(api_key=api_key)
    )
    nebuly_tracking_thread.daemon = True
    atexit.register(_stop_thread_when_main_ends, nebuly_tracking_thread)
    nebuly_tracking_thread.start()

    tracker_list = _instantiate_trackers(_nebuly_queue)
    for tracker in tracker_list:
        tracker.replace_sdk_functions()


@contextlib.contextmanager
def tracker(
    project: Optional[str] = None,
    phase: Optional[DevelopmentPhase] = None,
    task: Optional[Task] = None,
):
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
    _check_tracker_input_types(project, phase, task)

    if _nebuly_queue is None:
        raise RuntimeError("Please call nebuly.init() before using nebuly.tracker()")

    old_tag_data = copy.deepcopy(_nebuly_queue.tag_data)
    new_tag_data = TagData(
        project=project,
        phase=phase,
        task=task,
    )
    _nebuly_queue.patch_tag_data(new_tag_data)
    yield
    _nebuly_queue.patch_tag_data(old_tag_data)


def _instantiate_trackers(nebuly_queue: NebulyQueue) -> list:
    tracker_list = []
    try:
        from nebuly.trackers.openai import OpenAITracker

        tracker_list.append(OpenAITracker(nebuly_queue=nebuly_queue))
    except ImportError:
        pass

    return tracker_list


def _check_init_input_types(
    project: str, phase: DevelopmentPhase, task: Optional[Task]
):
    if isinstance(project, str) is False:
        raise TypeError(f"project must be of type str, not {type(project)}")
    if isinstance(phase, DevelopmentPhase) is False:
        raise TypeError(f"phase must be of type DevelopmentPhase, not {type(phase)}")
    if task is not None and isinstance(task, Task) is False:
        raise TypeError(f"task must be of type Task, not {type([task])}")


def _check_tracker_input_types(
    project: Optional[str], phase: Optional[DevelopmentPhase], task: Optional[Task]
):
    if project is not None and isinstance(project, str) is False:
        raise TypeError(f"project must be of type str, not {type(project)}")
    if phase is not None and isinstance(phase, DevelopmentPhase) is False:
        raise TypeError(f"phase must be of type DevelopmentPhase, not {type(phase)}")
    if task is not None and isinstance(task, Task) is False:
        raise TypeError(f"task must be of type Task, not {type([task])}")


def _stop_thread_when_main_ends(nebuly_thread_instance):
    if nebuly_thread_instance is not None:
        nebuly_thread_instance.thread_running = False
        nebuly_thread_instance.join()
