import copy
import time
from abc import ABC, abstractmethod
from queue import Empty, Queue
from threading import Thread
from typing import Any, Dict, Optional

from nebuly.core.clients import NebulyClient
from nebuly.core.schemas import (
    DevelopmentPhase,
    NebulyDataPackage,
    TagData,
    Task,
)
from nebuly.core.services import TaskDetector

QUEUE_MAX_SIZE = 10000


class Tracker(ABC):
    @abstractmethod
    def replace_sdk_functions(self) -> None:
        """Replaces the original functions of a provider
        with the new ones that can track required data.
        """
        raise NotImplementedError


class RawTrackedData(ABC):
    """Contains the raw data that is tracked by the SDK."""

    ...


class DataPackageConverter(ABC):
    def __init__(
        self,
        task_detector: TaskDetector = TaskDetector(),
    ) -> None:
        self._task_detector = task_detector

    @abstractmethod
    def get_data_package(
        self,
        raw_data: RawTrackedData,
        tag_data: TagData,
    ) -> NebulyDataPackage:
        """Converts the queue object to a data package.

        Args:
            raw_data (RawTrackedData): The data package info used to
                create the data package.
            tag_data (TagData): The data that contains information about
                project, development_phase and task.

        Returns:
            NebulyDataPackage: The data package.
        """
        raise NotImplementedError


class QueueObject:
    def __init__(
        self,
        raw_data: RawTrackedData,
        data_package_converter: DataPackageConverter,
    ) -> None:
        self._tag_data = TagData()
        self._raw_data = raw_data
        self._data_package_converter = data_package_converter

    def tag(self, tag_data: TagData) -> None:
        """Updates the tagged data.

        Args:
            tag_data (TagData): The tagged data.

        Raises:
            ValueError: If project name is not set.
            ValueError: If Development development_phase is not set.
        """
        if tag_data.project == "unknown":
            raise ValueError("Project name is not set.")
        if tag_data.development_phase == DevelopmentPhase.UNKNOWN:
            raise ValueError("Development development_phase is not set.")

        self._tag_data: TagData = self._clone(item=tag_data)

    def as_data_package(self) -> NebulyDataPackage:
        """Converts the queue object to a data package.
        It is used inside the NebulyThread to convert the raw data
        to a data package that is sent to Nebuly Server.
        To perform the conversion it uses the DataPacakgeConverter.

        Returns:
            NebulyDataPackage: The data package that is sent to Nebuly Server.
        """
        return self._data_package_converter.get_data_package(
            raw_data=self._raw_data,
            tag_data=self._tag_data,
        )

    @staticmethod
    def _clone(item: Any) -> Any:
        return copy.deepcopy(x=item)


class NebulyQueue(Queue):
    def __init__(
        self,
        tag_data: TagData,
        maxsize: int = QUEUE_MAX_SIZE,
    ) -> None:
        super().__init__(maxsize=maxsize)
        self.tag_data = tag_data

    def patch_tag_data(self, tag_data: TagData) -> None:
        """Patches the tagged data with the new tagged data.

        Args:
            tag_data (TagData): The new tagged data.
        """
        if tag_data.project != "unknown":
            self.tag_data.project = tag_data.project
        if tag_data.development_phase != DevelopmentPhase.UNKNOWN:
            self.tag_data.development_phase = tag_data.development_phase
        if tag_data.task != Task.UNKNOWN:
            self.tag_data.task = tag_data.task

    def put(
        self, item: QueueObject, block: bool = True, timeout: Optional[float] = None
    ) -> None:
        """Puts an QueueObject into the queue and tags it with the current
        tagged data.

        Args:
            item (QueueObject): The item to put into the queue.
            block (bool, optional): Whether to block the queue or not.
                Defaults to True.
            timeout (Optional[float], optional): The timeout for the queue.
                Defaults to None.
        """
        item.tag(tag_data=self.tag_data)
        super().put(item=item, block=block, timeout=timeout)

    def get(self, block: bool = True, timeout: Optional[float] = None) -> QueueObject:
        """Gets an QueueObject from the queue.

        Args:
            block (bool, optional): Whether to block the queue or not.
                Defaults to True.
            timeout (Optional[float], optional): The timeout for the queue.
                Defaults to None.
        """
        queue_object: QueueObject = super().get(block=block, timeout=timeout)
        return queue_object


class NebulyTrackingDataThread(Thread):
    def __init__(
        self,
        queue: NebulyQueue,
        nebuly_client: NebulyClient,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(**kwargs)

        self._queue = queue
        self._nebuly_client = nebuly_client
        self.thread_running = True
        self.force_exit = False

    def run(self) -> None:
        """Continuously takes elements from the queue and sends them to the
        Nebuly server.
        """
        while self.thread_running is True or self._queue.empty() is False:
            if self.force_exit is True:
                break

            try:
                queue_object = self._queue.get(timeout=0)
            except Empty:
                time.sleep(1)
                continue

            request_data = queue_object.as_data_package()
            self._nebuly_client.send_request_to_nebuly_server(request_data=request_data)
            self._queue.task_done()
