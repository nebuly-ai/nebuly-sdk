import atexit
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple

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

    @classmethod
    def from_torch_device(cls, device: torch.device) -> "Device":
        device_type = DeviceType.CPU
        device_id = None
        if "cuda" in device.type:
            device_type = DeviceType.GPU
            device_id = device.index

        return cls(device_type=device_type, device_id=device_id)

    @classmethod
    def from_str(cls, device_str: str) -> "Device":
        device_type = DeviceType.CPU
        device_id = None
        if "cuda" in device_str:
            device_type = DeviceType.GPU
            device_id = int(device_str.split(":")[1]) if ":" in device_str else 0

        return cls(device_type=device_type, device_id=device_id)


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
    device: Optional[Device] = None
    _start_event: Optional[torch.cuda.Event] = None
    end_event: Optional[torch.cuda.Event] = None
    end_time: Optional[float] = None

    @property
    def duration(self):
        if self.end_time is None:
            torch.cuda.synchronize()
            duration = self._start_event.elapsed_time(self.end_event) / 1000
            return duration
        else:
            return self.end_time - self.iteration_start_time


class PyTorchTracker(Tracker):
    def __init__(
        self, nebuly_queue: NebulyQueue, start_script_timestamp: float
    ) -> None:
        self._nebuly_queue: NebulyQueue = nebuly_queue
        self._start_script_timestamp = start_script_timestamp
        self._is_inside_forward = False
        self._is_training_started = False
        self._iteration_start_time = None
        self._training_step_cuda_storage: Dict[
            PyTorchJobOperation, PyTorchTrainingOpInfo
        ] = {}
        self._job_id = uuid.uuid4()

    def replace_sdk_functions(self) -> None:
        self._replace_data_loader()
        self._replace_forward_function()
        self._replace_backward_function()
        self._replace_optimizer_zero_grad_function()
        self._replace_optimizer_step_function()
        self._replace_tensor_move_to_device()

    @property
    def _has_cuda(self):
        return torch.cuda.is_available()

    @staticmethod
    def _is_model(model: torch.nn.Module) -> bool:
        if isinstance(model, _Loss):
            return False
        elif model.__class__.__name__ in torchvision.transforms.transforms.__all__:
            return False

        return True

    @staticmethod
    def _track_method_on_gpu(
        original_method: Callable, *args, **kwargs
    ) -> Tuple[Any, torch.cuda.Event, torch.cuda.Event]:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        out = original_method(*args, **kwargs)
        end_event.record()

        return out, start_event, end_event

    def _setup_start_events(self, start_timestamp: float):
        if not self._is_training_started:
            self._track_event(
                PyTorchJobOperation.START_SCRIPT, self._start_script_timestamp
            )
            atexit.register(self._track_event, PyTorchJobOperation.END_SCRIPT)
            self._track_event(PyTorchJobOperation.START_TRAINING, start_timestamp)
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

    def _track_event(
        self, job_operation: PyTorchJobOperation, timestamp: Optional[float] = None
    ):
        if timestamp is None:
            timestamp = time.time()
        data = PyTorchRawTrackedData(
            job_id=self._job_id,
            timestamp=timestamp,
            duration=None,
            device=None,
            job_mode=PyTorchJobMode.TRAINING,
            job_operation=job_operation,
        )

        self._track(raw_data=data)

    def _replace_tensor_move_to_device(self):
        original_to = torch.Tensor.to

        def new_to(tensor, *args, **kwargs):
            if (
                not self._has_cuda
                or len(args) < 1
                or not (
                    isinstance(args[0], torch.device)
                    or ("cuda" in args[0] or "cpu" in args[0])
                )
            ):
                tensor = original_to(tensor, *args, **kwargs)
            else:
                start_timestamp = time.time()
                tensor, start_event, end_event = self._track_method_on_gpu(
                    original_to, tensor, *args, **kwargs
                )

                if isinstance(args[0], torch.device):
                    device = Device.from_torch_device(args[0])
                else:
                    device = Device.from_str(args[0])

                if self._iteration_start_time is not None:
                    self._training_step_cuda_storage[
                        PyTorchJobOperation.MOVE_TO_DEVICE
                    ] = PyTorchTrainingOpInfo(
                        iteration_start_time=self._iteration_start_time,
                        _start_event=start_event,
                        end_event=end_event,
                        device=device,
                    )
                else:
                    torch.cuda.synchronize()
                    duration = start_event.elapsed_time(end_event) / 1000
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
                self._track_event(PyTorchJobOperation.END_TRAINING, end_training_time)
                raise StopIteration
            end_time = time.time()
            self._setup_start_events(start_timestamp)
            data = PyTorchRawTrackedData(
                job_id=self._job_id,
                timestamp=self._iteration_start_time,
                duration=end_time - self._iteration_start_time,
                device=None,
                job_mode=None,
                job_operation=PyTorchJobOperation.DATA_LOADER,
            )
            self._track(raw_data=data)

            return res

        setattr(torch.utils.data.dataloader._BaseDataLoaderIter, "__next__", new_iter)

    def _replace_forward_function(self) -> None:
        original_forward = torch.nn.Module.__call__

        def new_forward(model, *args, **kwargs):
            # Track the start of the training/iteration
            if model.training:
                self._setup_start_events(time.time())

            if self._is_model(model) and not self._is_inside_forward:
                self._is_inside_forward = True  # prevent internal calls to be tracked
                if not model.training:
                    start = time.time()
                else:
                    start = self._iteration_start_time

                # Get model device
                device = next(model.parameters()).device

                if self._has_cuda:
                    # GPU is available, we need to track the forward pass
                    # duration using CUDA events
                    out, start_event, end_event = self._track_method_on_gpu(
                        original_forward, model, *args, **kwargs
                    )

                    duration = None
                    if not model.training:
                        # Inference time, we need to wait for the end event and
                        # compute the duration
                        torch.cuda.synchronize()
                        duration = start_event.elapsed_time(end_event) / 1000
                    else:
                        # Training time, we can store the events and compute the
                        # duration later in the step function
                        self._training_step_cuda_storage[
                            PyTorchJobOperation.FORWARD
                        ] = PyTorchTrainingOpInfo(
                            iteration_start_time=start,
                            _start_event=start_event,
                            end_event=end_event,
                            device=Device.from_torch_device(device),
                        )
                else:
                    # GPU is not available, we can track the forward pass duration
                    # using time.time()
                    out = original_forward(model, *args, **kwargs)
                    duration = time.time() - start

                if not model.training or not self._has_cuda:
                    job_mode = (
                        PyTorchJobMode.TRAINING
                        if model.training
                        else PyTorchJobMode.EVALUATION
                    )
                    data = PyTorchRawTrackedData(
                        job_id=self._job_id,
                        timestamp=start,
                        duration=duration,
                        device=Device.from_torch_device(device),
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
            if self._has_cuda:
                out, start_event, end_event = self._track_method_on_gpu(
                    original_backward, tensor, *args, **kwargs
                )
                self._training_step_cuda_storage[
                    PyTorchJobOperation.BACKWARD
                ] = PyTorchTrainingOpInfo(
                    iteration_start_time=self._iteration_start_time,
                    _start_event=start_event,
                    end_event=end_event,
                    device=Device.from_torch_device(tensor.device),
                )
            else:
                out = original_backward(tensor, *args, **kwargs)
                end_time = time.time()
                job_mode = PyTorchJobMode.TRAINING
                data = PyTorchRawTrackedData(
                    job_id=self._job_id,
                    timestamp=self._iteration_start_time,
                    duration=end_time - self._iteration_start_time,
                    device=Device.from_torch_device(tensor.device),
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
            if self._has_cuda:
                _, start_event, end_event = self._track_method_on_gpu(
                    original_zero_grad, optimizer, *args, **kwargs
                )
                self._training_step_cuda_storage[
                    PyTorchJobOperation.OPTIMIZER_ZERO_GRAD
                ] = PyTorchTrainingOpInfo(
                    iteration_start_time=self._iteration_start_time,
                    _start_event=start_event,
                    end_event=end_event,
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
            if self._has_cuda:
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

        def _post_hook(*args, **kwargs):
            if self._has_cuda:
                end_event = torch.cuda.Event(enable_timing=True)
                end_event.record()
                self._training_step_cuda_storage[
                    PyTorchJobOperation.OPTIMIZER_STEP
                ].end_event = end_event
            else:
                end_time = time.time()
                self._training_step_cuda_storage[
                    PyTorchJobOperation.OPTIMIZER_STEP
                ].end_time = end_time

            job_mode = PyTorchJobMode.TRAINING

            for key, value in self._training_step_cuda_storage.items():
                data = PyTorchRawTrackedData(
                    job_id=self._job_id,
                    timestamp=value.iteration_start_time,
                    duration=value.duration,
                    device=value.device,
                    job_mode=job_mode,
                    job_operation=key,
                )
                self._track(raw_data=data)
            self._training_step_cuda_storage = {}
            self._iteration_start_time = None

        register_optimizer_step_pre_hook(_pre_hook)
        register_optimizer_step_post_hook(_post_hook)

    def _track(self, raw_data: PyTorchRawTrackedData) -> None:
        queue_object = PyTorchQueueObject(raw_data)
        self._nebuly_queue.put(item=queue_object, timeout=0)
