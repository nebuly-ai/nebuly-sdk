import atexit
import contextlib
import copy
from enum import Enum
from typing import Optional

from nebuly.core.schemas import Task, DevelopmentPhase, TagData
from nebuly.core.nebuly_client import NebulyQueue, NebulyTrackingDataThread


_nebuly_queue = None


def init(
    project: str,
    phase: DevelopmentPhase,
    task: Optional[Enum] = Task.UNDETECTED,
):
    _check_input_types(project, phase, task)

    global _nebuly_queue
    _nebuly_queue = NebulyQueue()
    tag_data = TagData(
        project=project,
        phase=phase,
        task=task,
    )
    _nebuly_queue.update_tagged_data(tag_data)
    _nebuly_queue.load_previous_status()

    nebuly_tracking_thread = NebulyTrackingDataThread(queue=_nebuly_queue)
    atexit.register(_stop_thread_when_main_ends, nebuly_tracking_thread)
    nebuly_tracking_thread.start()

    tracker_list = _instantiate_trackers(_nebuly_queue)
    for tracker in tracker_list:
        tracker.replace_sdk_functions()


@contextlib.contextmanager
def tracker(
    project: Optional[str] = None,
    phase: Optional[DevelopmentPhase] = None,
    task: Optional[Enum] = None,
):
    _check_input_types(project, phase, task)

    if _nebuly_queue is None:
        raise RuntimeError("Please call nebuly.init() before using nebuly.tracker()")

    old_tag_data = copy.deepcopy(_nebuly_queue.tagged_data)
    new_tag_data = TagData(
        project=project,
        phase=phase,
        task=task,
    )
    _nebuly_queue.update_tagged_data(new_tag_data)
    print(_nebuly_queue.tagged_data)
    yield
    _nebuly_queue.update_tagged_data(old_tag_data)


def _instantiate_trackers(nebuly_queue: NebulyQueue) -> list:
    tracker_list = []
    try:
        from nebuly.trackers.openai import OpenAITracker

        tracker_list.append(OpenAITracker(nebuly_queue=nebuly_queue))
    except ImportError:
        pass

    return tracker_list


def _check_input_types(project: str, phase: DevelopmentPhase, task: Task):
    if isinstance(project, str) is False:
        raise TypeError(f"project must be of type str, not {type(project)}")
    if isinstance(phase, DevelopmentPhase) is False:
        raise TypeError(f"phase must be of type DevelopmentPhase, not {type(phase)}")
    if isinstance(task, Task) is False:
        raise TypeError(f"task must be of type Task, not {type(task)}")


def _stop_thread_when_main_ends(nebuly_thread_instance):
    if nebuly_thread_instance is not None:
        nebuly_thread_instance.thread_running = False
        nebuly_thread_instance.join()
