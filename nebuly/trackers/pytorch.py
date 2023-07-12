import time
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import Optional

import torch
import torchvision
from torch.nn.modules.loss import _Loss

from nebuly.core.queues import (
    DataPackageConverter,
    NebulyQueue,
    QueueObject,
    RawTrackedData,
    Tracker,
)
from nebuly.core.schemas import (
    GenericProviderAttributes,
    NebulyDataPackage,
    Provider,
    TagData,
)


class PyTorchJobMode(Enum):
    TRAINING = "training"
    EVALUATION = "evaluation"


class PyTorchJobOperation(Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    OPTIMIZER_STEP = "optimizer_step"
    OPTIMIZER_ZERO_GRAD = "optimizer_zero_grad"
    DATA_LOADER = "data_loader"
    START_TRAINING = "start_training"
    END_TRAINING = "end_training"
    START_SCRIPT = "start_script"
    END_SCRIPT = "end_script"


class DeviceType(Enum):
    CPU = "cpu"
    GPU = "gpu"


@dataclass
class Device:
    device_type: DeviceType
    device_id: Optional[int]


@dataclass
class PyTorchRawTrackedData(RawTrackedData):
    timestamp: float
    device: Device
    job_mode: PyTorchJobMode
    job_operation: PyTorchJobOperation
    timestamp_end: Optional[float] = None


class PyTorchAttributes(GenericProviderAttributes):
    job_mode: PyTorchJobMode
    job_operation: PyTorchJobOperation
    device: Device


class PyTorchDataPackageConverter(DataPackageConverter):
    def get_data_package(
        self,
        raw_data: PyTorchRawTrackedData,
        tag_data: TagData,
    ) -> NebulyDataPackage:
        provider = Provider.PYTORCH
        body = PyTorchAttributes(
            project=tag_data.project,
            phase=tag_data.phase,
            task=tag_data.task,
            timestamp=raw_data.timestamp,
            timestamp_end=raw_data.timestamp_end,
            job_mode=raw_data.job_mode,
            job_operation=raw_data.job_operation,
            device=raw_data.device,
        )
        return NebulyDataPackage(kind=provider, body=body)


class PyTorchQueueObject(QueueObject):
    def __init__(
        self,
        raw_data: PyTorchRawTrackedData,
        data_package_converter: DataPackageConverter = PyTorchDataPackageConverter(),
    ) -> None:
        super().__init__(
            raw_data=raw_data, data_package_converter=data_package_converter
        )


@dataclass
class PyTorchTrainingOpInfo:
    start_time: float
    _start_event: Optional[torch.cuda.Event]
    _end_event: Optional[torch.cuda.Event]

    @property
    def end_time(self):
        torch.cuda.synchronize()
        duration = self._end_event.elapsed_time(self._start_event)
        return self.start_time + duration


class PyTorchTracker(Tracker):
    def __init__(self, nebuly_queue: NebulyQueue) -> None:
        self._nebuly_queue: NebulyQueue = nebuly_queue
        self._is_inside_forward = False
        self._training_step_cuda_storage = {}

    @cached_property
    def has_cuda(self):
        return torch.cuda.is_available()

    @staticmethod
    def is_model(model: torch.nn.Module) -> bool:
        if isinstance(model, _Loss):
            return False
        elif model.__class__.__name__ in torchvision.transforms.transforms.__all__:
            return False

        return True

    def replace_sdk_functions(self) -> None:
        # self._replace_forward_function()
        self._replace_backward_function()

    def _replace_forward_function(self) -> None:
        original_forward = torch.nn.Module.__call__

        def new_forward(model, *args, **kwargs):
            if self.is_model(model) and not self._is_inside_forward:
                self._is_inside_forward = True
                start = time.time()
                end = None
                # If the model is on GPU:
                if self.has_cuda:
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    out = original_forward(model, *args, **kwargs)
                    end_event.record()

                    if not model.training:
                        torch.cuda.synchronize()
                        end = start + start_event.elapsed_time(end_event)
                    else:
                        self._training_step_cuda_storage[
                            PyTorchJobOperation.FORWARD
                        ] = PyTorchTrainingOpInfo(
                            start_time=start,
                            _start_event=start_event,
                            _end_event=end_event,
                        )
                else:
                    out = original_forward(model, *args, **kwargs)
                    end = time.time()

                if not (model.training and self.has_cuda):
                    device = next(model.parameters()).device
                    device_type = (
                        DeviceType.GPU if device.type == "cuda" else DeviceType.CPU
                    )
                    job_mode = (
                        PyTorchJobMode.TRAINING
                        if model.training
                        else PyTorchJobMode.EVALUATION
                    )
                    data = PyTorchRawTrackedData(
                        timestamp=start,
                        timestamp_end=end,
                        device=Device(device_type=device_type, device_id=device.index),
                        job_mode=job_mode,
                        job_operation=PyTorchJobOperation.FORWARD,
                    )
                    self._track(raw_data=data)
                self._is_inside_forward = False
            else:
                out = original_forward(model, *args, **kwargs)

            return out

        setattr(torch.nn.Module, "__call__", new_forward)

    def _replace_backward_function(self):
        original_backward = torch.Tensor.backward

        def new_backward(tensor, *args, **kwargs):
            start_time = time.time()
            if self.has_cuda:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                out = original_backward(tensor, *args, **kwargs)
                end_event.record()
                self._training_step_cuda_storage[
                    PyTorchJobOperation.BACKWARD
                ] = PyTorchTrainingOpInfo(
                    start_time=start_time,
                    _start_event=start_event,
                    _end_event=end_event,
                )
            else:
                out = original_backward(tensor, *args, **kwargs)
                end_time = time.time()
                device = tensor.device
                device_type = (
                    DeviceType.GPU if device.type == "cuda" else DeviceType.CPU
                )
                job_mode = PyTorchJobMode.TRAINING
                data = PyTorchRawTrackedData(
                    timestamp=start_time,
                    timestamp_end=end_time,
                    device=Device(device_type=device_type, device_id=device.index),
                    job_mode=job_mode,
                    job_operation=PyTorchJobOperation.BACKWARD,
                )
                self._track(raw_data=data)
            return out

        setattr(torch.Tensor, "backward", new_backward)

    def _track(self, raw_data: PyTorchRawTrackedData) -> None:
        queue_object = PyTorchQueueObject(raw_data)
        self._nebuly_queue.put(item=queue_object, timeout=0)
