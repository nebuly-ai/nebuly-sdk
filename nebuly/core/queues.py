import copy
from abc import ABC, abstractmethod
from queue import Queue
from typing import Any, Optional

from nebuly.core.schemas import (
    DevelopmentPhase,
    NebulyDataPackage,
    Task,
    TagData,
)
from nebuly.core.services import TaskDetector

QUEUE_MAX_SIZE = 10000


class Tracker(ABC):
    @abstractmethod
    def replace_sdk_functions(self) -> None:
        pass


class DataPackageConverter(ABC):
    def __init__(
        self,
        task_detector: TaskDetector = TaskDetector(),
    ) -> None:
        self._task_detector = task_detector

    @abstractmethod
    def get_data_package(
        self,
        tag_data: TagData,
        timestamp: float,
        timestamp_end: float,
    ) -> NebulyDataPackage:
        """Converts the queue object to a data package.

        Args:
            tag_data (TagData): The tagged data contained the user specified-tags.
            timestamp (float): The timestamp captured at the beginning of the request.
            timestamp_end (float): The timestamp captured at the end of the request.

        Returns:
            NebulyDataPackage: The data package.
        """
        pass


class QueueObject(ABC):
    def __init__(
        self,
    ) -> None:
        self._tag_data = TagData()

    def tag(self, tag_data: TagData) -> None:
        """Updates the tagged data.

        Args:
            tag_data (TagData): The tagged data.

        Raises:
            ValueError: If project name is not set.
            ValueError: If Development phase is not set.
        """
        if tag_data.project == "unknown":
            raise ValueError("Project name is not set.")
        if tag_data.phase == DevelopmentPhase.UNKNOWN:
            raise ValueError("Development phase is not set.")

        self._tag_data: TagData = self._clone(item=tag_data)

    @abstractmethod
    def as_data_package(self) -> NebulyDataPackage:
        """Converts the queue object to a data package.

        Returns:
            NebulyDataPackage: The data package.
        """
        pass

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
        if tag_data.phase != DevelopmentPhase.UNKNOWN:
            self.tag_data.phase = tag_data.phase
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
