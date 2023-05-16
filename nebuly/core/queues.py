from abc import ABC, abstractmethod
from queue import Queue
from typing import Dict, Optional, Any
import time

from nebuly.core.schemas import (
    DevelopmentPhase,
    Task,
    TagData,
    NebulyDataPackage,
    Provider,
)
from nebuly.core.services import TaskDetector


class DataPackageConverter(ABC):
    def __init__(
        self,
        task_detector: TaskDetector = TaskDetector(),
    ):
        self._tagged_data = TagData(
            project=None,
            phase=None,
            task=None,
        )
        self._task_detector = task_detector
        self._api_type = None
        self._provider = Provider.UNKNOWN

    def load_tag_data(self, tagged_data: TagData):
        """Updates the tagged data.

        Args:
            tagged_data (TagData): The tagged data.
        """
        self._tagged_data.project = tagged_data.project
        self._tagged_data.phase = tagged_data.phase
        self._tagged_data.task = tagged_data.task

    def load_request_data(
        self,
        request_kwargs: Dict,
        request_response: Dict,
        api_type: Any,
        timestamp: float,
    ):
        """Loads the request data.

        Args:
            request_kwargs (Dict): The request kwargs.
            request_response (Dict): The request response.
        """
        self._request_kwargs = request_kwargs
        self._request_response = request_response
        self._timestamp = timestamp
        self._api_type = api_type

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
        request_kwargs: Dict,
        request_response: Dict,
        api_type: Any,
        data_package_converter: DataPackageConverter,
    ):
        self._request_kwargs = request_kwargs
        self._request_response = request_response
        self._api_type = api_type
        self._data_package_converter = data_package_converter

        self._tagged_data = TagData(
            project=None,
            phase=None,
            task=None,
        )

        self._timestamp = time.time()

    def tag(self, tagged_data: TagData):
        """Updates the tagged data.

        Args:
            tagged_data (TagData): The tagged data.

        Raises:
            ValueError: If project name is not set.
            ValueError: If development phase is not set.
        """
        if tagged_data.project is None:
            raise ValueError("Project name is not set.")
        if tagged_data.phase is None:
            raise ValueError("Development phase is not set.")

        self._tagged_data.project = tagged_data.project
        self._tagged_data.phase = tagged_data.phase
        self._tagged_data.task = tagged_data.task

    def as_data_package(self) -> NebulyDataPackage:
        """Converts the queue object to a data package.

        Returns:
            NebulyDataPackage: The data package.
        """
        self._data_package_converter.load_tag_data(self._tagged_data)
        self._data_package_converter.load_request_data(
            self._request_kwargs,
            self._request_response,
            self._api_type,
            self._timestamp,
        )
        return self._data_package_converter.get_data_package()


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

    def patch_tagged_data(self, new_tag_data: TagData):
        """Patches the tagged data with the new tagged data.

        Args:
            new_tag_data (TagData): The new tagged data.
        """
        if new_tag_data.project is not None:
            self.tagged_data.project = new_tag_data.project
        if new_tag_data.phase is not None:
            self.tagged_data.phase = new_tag_data.phase
        if new_tag_data.task is not None:
            self.tagged_data.task = new_tag_data.task

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
        item.tag(self.tagged_data)
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
