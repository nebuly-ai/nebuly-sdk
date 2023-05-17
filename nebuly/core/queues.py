from abc import ABC, abstractmethod
import copy
from queue import Queue
from typing import Optional

from nebuly.core.schemas import (
    DevelopmentPhase,
    Task,
    TagData,
    NebulyDataPackage,
)
from nebuly.core.services import TaskDetector


class DataPackageConverter(ABC):
    def __init__(
        self,
        task_detector: TaskDetector = TaskDetector(),
    ):
        self.tag_data = TagData(
            project=None,
            phase=None,
            task=None,
        )
        self._task_detector = task_detector

    @abstractmethod
    def get_data_package(self) -> NebulyDataPackage:
        """Converts the queue object to a data package.

        Returns:
            NebulyDataPackage: The data package.
        """
        pass


class QueueObject:
    def __init__(
        self,
        data_package_converter: DataPackageConverter,
    ):
        self._data_package_converter = data_package_converter
        self._tag_data = TagData(
            project=None,
            phase=None,
            task=None,
        )

    def tag(self, tag_data: TagData):
        """Updates the tagged data.

        Args:
            tag_data (TagData): The tagged data.

        Raises:
            ValueError: If project name is not set.
            ValueError: If development phase is not set.
        """
        if tag_data.project is None:
            raise ValueError("Project name is not set.")
        if tag_data.phase is None:
            raise ValueError("Development phase is not set.")

        self._tag_data = self._clone(tag_data)

    def as_data_package(self) -> NebulyDataPackage:
        """Converts the queue object to a data package.

        Returns:
            NebulyDataPackage: The data package.
        """
        self._data_package_converter.tag_data = self._tag_data
        return self._data_package_converter.get_data_package()

    @staticmethod
    def _clone(item):
        return copy.deepcopy(item)


class NebulyQueue(Queue):
    def __init__(
        self,
    ):
        super().__init__()
        self.tag_data = TagData(
            project="unknown_project",
            phase=DevelopmentPhase.UNKNOWN,
            task=Task.UNDETECTED,
        )

    def patch_tag_data(self, tag_data: TagData):
        """Patches the tagged data with the new tagged data.

        Args:
            new_tag_data (TagData): The new tagged data.
        """
        if tag_data.project is not None:
            self.tag_data.project = tag_data.project
        if tag_data.phase is not None:
            self.tag_data.phase = tag_data.phase
        if tag_data.task is not None:
            self.tag_data.task = tag_data.task

    def put(
        self, item: QueueObject, block: bool = True, timeout: Optional[float] = None
    ):
        """Puts an QueueObject into the queue and tags it with the current
        tagged data.

        Args:
            item (QueueObject): The item to put into the queue.
            block (bool, optional): Whether to block the queue or not.
                Defaults to True.
            timeout (Optional[float], optional): The timeout for the queue.
                Defaults to None.
        """
        item.tag(self.tag_data)
        super().put(item=item, block=block, timeout=timeout)

    def get(self, block: bool = True, timeout: Optional[float] = None) -> QueueObject:
        """Gets an QueueObject from the queue.

        Args:
            block (bool, optional): Whether to block the queue or not.
                Defaults to True.
            timeout (Optional[float], optional): The timeout for the queue.
                Defaults to None.
        """
        queue_object = super().get(block=block, timeout=timeout)
        return queue_object

    def save_current_status(self):
        # Not implemented yet.
        # save the current status of the queue to disk
        # required when the program is interrupted and must resume
        # without losing data
        pass

    def load_previous_status(self):
        # Not implemented yet.
        # load the previous status of the queue from disk
        # required when the program is interrupted and must resume
        # without losing data
        pass
