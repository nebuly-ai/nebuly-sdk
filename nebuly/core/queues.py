from abc import ABC, abstractmethod
import copy
from queue import Queue
from typing import Optional, Dict

from nebuly.core.schemas import (
    DevelopmentPhase,
    Task,
    TagData,
    NebulyDataPackage,
    NebulyRequestParams,
)
from nebuly.core.services import TaskDetector


QUEUE_MAX_SIZE = 10000


class DataPackageConverter(ABC):
    def __init__(
        self,
        task_detector: TaskDetector = TaskDetector(),
    ):
        self._task_detector = task_detector

    @abstractmethod
    def get_data_package(
        self,
        tag_data: TagData,
        request_kwargs: Dict,
        request_response: Dict,
        api_type: str,
        timestamp: float,
    ) -> NebulyDataPackage:
        """Converts the queue object to a data package.

        Args:
            tag_data (TagData): The tagged data contained the user specified-tags.
            request_kwargs (Dict): The request kwargs.
            request_response (Dict): The request response.
            api_type (str): The api type.
            timestamp (float): The timestamp of the tracked request.

        Returns:
            NebulyDataPackage: The data package.
        """
        pass

    @abstractmethod
    def get_request_params(self) -> NebulyRequestParams:
        """Returns the request params.

        Returns:
            NebulyRequestParams: The request params.
        """


class QueueObject(ABC):
    def __init__(
        self,
    ):
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

    @abstractmethod
    def as_data_package(self) -> NebulyDataPackage:
        """Converts the queue object to a data package.

        Returns:
            NebulyDataPackage: The data package.
        """
        pass

    @abstractmethod
    def as_request_params(self) -> NebulyRequestParams:
        """Returns the request params.

        Returns:
            NebulyRequestParams: The request params.
        """
        pass

    @staticmethod
    def _clone(item):
        return copy.deepcopy(item)


class NebulyQueue(Queue):
    def __init__(
        self,
    ):
        super().__init__(maxsize=QUEUE_MAX_SIZE)
        self.tag_data = TagData(
            project="unknown_project",
            phase=DevelopmentPhase.UNKNOWN,
            task=Task.UNDETECTED,
        )

    def patch_tag_data(self, tag_data: TagData):
        """Patches the tagged data with the new tagged data.

        Args:
            tag_data (TagData): The new tagged data.
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
