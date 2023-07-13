import time
import uuid
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import Dict, Optional

import torch
import torchvision
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import (
    register_optimizer_step_post_hook,
    register_optimizer_step_pre_hook,
)

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
    START_ITERATION = "start_iteration"
    MOVE_TO_DEVICE = "move_to_device"


class DeviceType(Enum):
    CPU = "cpu"
    GPU = "gpu"


@dataclass
class Device:
    device_type: DeviceType
    device_id: Optional[int]


@dataclass
class PyTorchRawTrackedData(RawTrackedData):
    job_id: uuid.UUID
    timestamp: float
    device: Optional[Device]
    job_mode: Optional[PyTorchJobMode]
    job_operation: PyTorchJobOperation
    duration: Optional[float] = None


class PyTorchAttributes(GenericProviderAttributes):
    job_mode: Optional[PyTorchJobMode]
    job_operation: Optional[PyTorchJobOperation]
    device: Optional[Device]
    duration: Optional[float]


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
            duration=raw_data.duration,
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
    iteration_start_time: float
    _start_event: Optional[torch.cuda.Event] = None
    _end_event: Optional[torch.cuda.Event] = None
    _end_time: Optional[float] = None

    @property
    def duration(self):
        if self._end_time is None:
            torch.cuda.synchronize()
            duration = self._start_event.elapsed_time(self._end_event) / 1000
            return duration
        else:
            return self._end_time - self.iteration_start_time


class PyTorchTracker(Tracker):
    def __init__(self, nebuly_queue: NebulyQueue) -> None:
        self._nebuly_queue: NebulyQueue = nebuly_queue
        self._is_inside_forward = False
        self._is_training_started = False
        self._iteration_start_time = None
        self._training_step_cuda_storage: Dict[
            PyTorchJobOperation, PyTorchTrainingOpInfo
        ] = {}
        self._job_id = uuid.uuid4()

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
        self._replace_data_loader()
        self._replace_forward_function()
        self._replace_backward_function()
        self._replace_optimizer_zero_grad_function()
        self._replace_optimizer_step_function()
        self._replace_tensor_move_to_device()

    def _setup_start_events(self, start_timestamp: float):
        if not self._is_training_started:
            self._track_start_training(start_timestamp)
            self._is_training_started = True
        if self._iteration_start_time is None:
            self._iteration_start_time = start_timestamp
            data = PyTorchRawTrackedData(
                job_id=self._job_id,
                timestamp=self._iteration_start_time,
                duration=None,
                device=None,
                job_mode=None,
                job_operation=PyTorchJobOperation.START_ITERATION,
            )
            self._track(raw_data=data)

    def _replace_tensor_move_to_device(self):
        original_to = torch.Tensor.to

        def new_to(tensor, *args, **kwargs):
            if (
                not self.has_cuda
                or len(args) < 1
                or not (
                    isinstance(args[0], torch.device)
                    or ("cuda" in args[0] or "cpu" in args[0])
                )
            ):
                tensor = original_to(tensor, *args, **kwargs)
            else:
                start_timestamp = time.time()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                tensor = original_to(tensor, *args, **kwargs)
                end_event.record()
                if self._iteration_start_time is not None:
                    self._training_step_cuda_storage[
                        PyTorchJobOperation.MOVE_TO_DEVICE
                    ] = PyTorchTrainingOpInfo(
                        iteration_start_time=self._iteration_start_time,
                        _start_event=start_event,
                        _end_event=end_event,
                    )
                else:
                    torch.cuda.synchronize()
                    duration = start_event.elapsed_time(end_event) / 1000
                    if isinstance(args[0], torch.device):
                        device_type = (
                            DeviceType.GPU if "cuda" in args[0].type else DeviceType.CPU
                        )
                        index = args[0].index
                    else:
                        device_type = (
                            DeviceType.GPU if "cuda" in args[0] else DeviceType.CPU
                        )
                        index_split_list = args[0].split(":")
                        index = None
                        if len(index_split_list) > 1:
                            index = int(index_split_list[1])
                        elif device_type is DeviceType.GPU:
                            index = 0
                    device = Device(device_type=device_type, device_id=index)

                    data = PyTorchRawTrackedData(
                        job_id=self._job_id,
                        timestamp=start_timestamp,
                        duration=duration,
                        device=device,
                        job_mode=None,
                        job_operation=PyTorchJobOperation.MOVE_TO_DEVICE,
                    )
                    self._track(raw_data=data)

            return tensor

        torch.Tensor.to = new_to

    def _replace_data_loader(self):
        original_iter = torch.utils.data.dataloader._BaseDataLoaderIter.__next__

        def new_iter(data_loader, *args, **kwargs):
            start_timestamp = time.time()
            try:
                res = original_iter(data_loader, *args, **kwargs)
            except StopIteration:
                end_training_time = time.time()
                self._track_end_training(end_training_time)
                raise StopIteration
            end_time = time.time()
            self._setup_start_events(start_timestamp)
            device = None
            job_mode = None
            data = PyTorchRawTrackedData(
                job_id=self._job_id,
                timestamp=self._iteration_start_time,
                duration=end_time - self._iteration_start_time,
                device=device,
                job_mode=job_mode,
                job_operation=PyTorchJobOperation.DATA_LOADER,
            )
            self._track(raw_data=data)

            return res

        setattr(torch.utils.data.dataloader._BaseDataLoaderIter, "__next__", new_iter)

    def _replace_forward_function(self) -> None:
        original_forward = torch.nn.Module.__call__

        def new_forward(model, *args, **kwargs):
            # Track the start orf the training
            if model.training:
                self._setup_start_events(time.time())

            if self.is_model(model) and not self._is_inside_forward:
                self._is_inside_forward = True
                if not model.training:
                    start = time.time()
                else:
                    start = self._iteration_start_time

                # If the model is on GPU:
                if self.has_cuda:
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    out = original_forward(model, *args, **kwargs)
                    end_event.record()

                    if not model.training:
                        torch.cuda.synchronize()
                        duration = start_event.elapsed_time(end_event) / 1000
                    else:
                        self._training_step_cuda_storage[
                            PyTorchJobOperation.FORWARD
                        ] = PyTorchTrainingOpInfo(
                            iteration_start_time=start,
                            _start_event=start_event,
                            _end_event=end_event,
                        )
                else:
                    out = original_forward(model, *args, **kwargs)
                    duration = time.time() - start

                if not model.training or not self.has_cuda:
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
                        job_id=self._job_id,
                        timestamp=start,
                        duration=duration,
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
            if self.has_cuda:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                out = original_backward(tensor, *args, **kwargs)
                end_event.record()
                self._training_step_cuda_storage[
                    PyTorchJobOperation.BACKWARD
                ] = PyTorchTrainingOpInfo(
                    iteration_start_time=self._iteration_start_time,
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
                    job_id=self._job_id,
                    timestamp=self._iteration_start_time,
                    duration=end_time - self._iteration_start_time,
                    device=Device(device_type=device_type, device_id=device.index),
                    job_mode=job_mode,
                    job_operation=PyTorchJobOperation.BACKWARD,
                )
                self._track(raw_data=data)
            return out

        setattr(torch.Tensor, "backward", new_backward)

    def _replace_optimizer_zero_grad_function(self):
        original_zero_grad = torch.optim.Optimizer.zero_grad

        def new_zero_grad(optimizer, *args, **kwargs):
            self._setup_start_events(time.time())
            if self.has_cuda:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                original_zero_grad(optimizer, *args, **kwargs)
                end_event.record()
                self._training_step_cuda_storage[
                    PyTorchJobOperation.OPTIMIZER_ZERO_GRAD
                ] = PyTorchTrainingOpInfo(
                    iteration_start_time=self._iteration_start_time,
                    _start_event=start_event,
                    _end_event=end_event,
                )
            else:
                original_zero_grad(optimizer, *args, **kwargs)
                end_time = time.time()
                job_mode = PyTorchJobMode.TRAINING
                data = PyTorchRawTrackedData(
                    job_id=self._job_id,
                    timestamp=self._iteration_start_time,
                    duration=end_time - self._iteration_start_time,
                    device=None,
                    job_mode=job_mode,
                    job_operation=PyTorchJobOperation.OPTIMIZER_ZERO_GRAD,
                )
                self._track(raw_data=data)

        torch.optim.Optimizer.zero_grad = new_zero_grad

    def _replace_optimizer_step_function(self):
        def _pre_hook(*args, **kwargs):
            if self.has_cuda:
                start_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                self._training_step_cuda_storage[
                    PyTorchJobOperation.OPTIMIZER_STEP
                ] = PyTorchTrainingOpInfo(
                    iteration_start_time=self._iteration_start_time,
                    _start_event=start_event,
                )
            else:
                self._training_step_cuda_storage[
                    PyTorchJobOperation.OPTIMIZER_STEP
                ] = PyTorchTrainingOpInfo(
                    iteration_start_time=self._iteration_start_time,
                )

        def _post_hook(model, *args, **kwargs):
            if self.has_cuda:
                end_event = torch.cuda.Event(enable_timing=True)
                end_event.record()
                self._training_step_cuda_storage[
                    PyTorchJobOperation.OPTIMIZER_STEP
                ]._end_event = end_event
            else:
                end_time = time.time()
                self._training_step_cuda_storage[
                    PyTorchJobOperation.OPTIMIZER_STEP
                ]._end_time = end_time

            job_mode = PyTorchJobMode.TRAINING

            for key, value in self._training_step_cuda_storage.items():
                data = PyTorchRawTrackedData(
                    job_id=self._job_id,
                    timestamp=value.iteration_start_time,
                    duration=value.duration,
                    device=None,
                    job_mode=job_mode,
                    job_operation=key,
                )
                self._track(raw_data=data)
            self._training_step_cuda_storage = {}
            self._iteration_start_time = None

        register_optimizer_step_pre_hook(_pre_hook)
        register_optimizer_step_post_hook(_post_hook)

    def _track_start_training(self, start_timestamp: float):
        data = PyTorchRawTrackedData(
            job_id=self._job_id,
            timestamp=start_timestamp,
            duration=None,
            device=None,
            job_mode=PyTorchJobMode.TRAINING,
            job_operation=PyTorchJobOperation.START_TRAINING,
        )

        self._track(raw_data=data)

    def _track_end_training(self, end_timestamp: float):
        data = PyTorchRawTrackedData(
            job_id=self._job_id,
            timestamp=end_timestamp,
            duration=None,
            device=None,
            job_mode=None,
            job_operation=PyTorchJobOperation.END_TRAINING,
        )
        self._track(raw_data=data)

    def _track(self, raw_data: PyTorchRawTrackedData) -> None:
        queue_object = PyTorchQueueObject(raw_data)
        # print(queue_object._raw_data)
        self._nebuly_queue.put(item=queue_object, timeout=0)
