from abc import ABC, abstractmethod
import logging
from queue import Queue, Empty
from typing import Dict
from threading import Thread
import time

from nebuly.core.schemas import DevelopmentPhase, Task, TagData, NebulyDataPackage
from nebuly.utils.task_detector import TaskDetector


class QueueObject(ABC, TaskDetector):
    def __init__(
        self,
        parameters: Dict,
        response: Dict,
    ):
        ABC.__init__(self)
        TaskDetector.__init__(self)

        self._project = None
        self._phase = None
        self._task = None

        self._parameters = parameters
        self._response = response
        self._timestamp = time.time()

    @abstractmethod
    def get_request_data(self) -> NebulyDataPackage:
        if self._project is None:
            raise RuntimeError(
                "You must place the QueueObject in the Queue "
                "before calling the unpack method"
            )


class NebulyQueue(Queue):
    def __init__(
        self,
    ):
        super().__init__()
        self.tagged_data = TagData(
            project="unknown_project",
            phase=DevelopmentPhase.UNKNOWN,
            task=Task.UNDETECTED,
        )

    def update_tagged_data(self, new_tag_data: TagData):
        if new_tag_data.project is not None:
            self.tagged_data.project = new_tag_data.project
        if new_tag_data.phase is not None:
            self.tagged_data.phase = new_tag_data.phase
        if new_tag_data.task is not None:
            self.tagged_data.task = new_tag_data.task

    def put(self, queue_object: QueueObject):
        queue_object._project = self.tagged_data.project
        queue_object._phase = self.tagged_data.phase
        queue_object._task = self.tagged_data.task
        super().put(queue_object)

    def save_current_status(self):
        pass

    def load_previous_status(self):
        pass


class NebulyClient:
    def __init__(self):
        pass

    def send_request_to_nebuly_server(self, request_data: NebulyDataPackage):
        logging.info(f"\n\nDetected Data:\n {request_data.json()}")


class NebulyTrackingDataThread(Thread):
    def __init__(
        self,
        queue: NebulyQueue,
        nebuly_client: NebulyClient = NebulyClient(),
    ):
        super().__init__()
        self.deamon = True

        self._queue = queue
        self._nebuly_client = nebuly_client
        self._thread_running = True

    def run(self):
        while self._thread_running:
            try:
                queue_object = self._queue.get()
            except Empty:
                continue
            except KeyboardInterrupt:
                # what happens if there is a keyboard interrupt?
                # I need to save the status of the queue and load it
                # back when it is started again
                pass

            request_data = queue_object.get_request_data()
            self._nebuly_client.send_request_to_nebuly_server(request_data)
            self._queue.task_done()
