import gc
from enum import Enum
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from threading import BoundedSemaphore

import cv2.typing

from utils import logger
from cascade import EventElement


@dataclass
class _CaptureImageData:
    img_data: cv2.typing.MatLike | None = None
    timestamp: datetime | None = None


@dataclass
class _CascadeResultData:
    event_element: EventElement | None = None
    total_runtime: float | None = None
    overhead: float | None = None


class BufferState(Enum):
    WAITING_FRAME = 1
    WAITING_CASCADE = 2
    WAITING_AGGREGATION = 3
    USED = 4


class ImageContainer:
    def __init__(self):
        self.capture_data: _CaptureImageData = _CaptureImageData()
        self.casc_result_data: _CascadeResultData = _CascadeResultData()
        self.img_semaphore = BoundedSemaphore(1)
        self.casc_compute_semaphore = BoundedSemaphore(1)
        self.casc_result_avail_semaphore = BoundedSemaphore(1)
        self.casc_compute_semaphore.acquire()
        self.casc_result_avail_semaphore.acquire()
        self.buffer_state = BufferState.WAITING_FRAME

    def __del__(self) -> None:
        """
        Cleans the data in this image container.
        """
        self.capture_data: _CaptureImageData
        self.casc_result_data: _CascadeResultData
        self.img_semaphore = BoundedSemaphore(1)
        self.casc_compute_semaphore = BoundedSemaphore(1)
        self.casc_result_avail_semaphore = BoundedSemaphore(1)
        self.casc_compute_semaphore.acquire()
        self.casc_result_avail_semaphore.acquire()

    # Methods to check the state of the locks of this buffer

    def is_ready_for_cascade(self) -> bool:
        return self.buffer_state == BufferState.WAITING_CASCADE

    def is_ready_for_aggregation(self) -> bool:
        return self.buffer_state == BufferState.WAITING_AGGREGATION

    def has_been_used(self) -> bool:
        return self.buffer_state == BufferState.USED

    # Methods to acquire and release the lock of this buffer

    def try_acquire_img_lock(self) -> bool:
        return self.img_semaphore.acquire(blocking=False)

    def try_acquire_casc_compute_lock(self) -> bool:
        return self.casc_compute_semaphore.acquire(blocking=False)

    def try_acquire_casc_result_available_lock(self) -> bool:
        return self.casc_result_avail_semaphore.acquire(blocking=False)

    def release_img_lock(self) -> None:
        self.img_semaphore.release()

    def release_casc_compute_lock(self) -> None:
        self.buffer_state = BufferState.WAITING_CASCADE
        self.casc_compute_semaphore.release()

    def release_casc_res_available_lock(self) -> None:
        self.buffer_state = BufferState.WAITING_AGGREGATION
        self.casc_result_avail_semaphore.release()

    # Methods used to store the data in this buffer

    def write_capture_data(self, img_data, timestamp: datetime) -> None:
        self.capture_data = _CaptureImageData(img_data, timestamp)

    def write_cascade_data(self, event_elem: EventElement, total_time: float, overhead: float):
        self.casc_result_data = _CascadeResultData(event_elem, total_time, overhead)

    # Accessors for the data stored in this buffer

    def get_img_data(self):
        return self.capture_data.img_data

    def get_timestamp(self) -> datetime:
        return self.capture_data.timestamp

    def get_event_element(self) -> EventElement:
        return self.casc_result_data.event_element

    def get_total_runtime(self) -> float:
        return self.casc_result_data.total_runtime

    def get_overhead(self) -> float:
        return self.casc_result_data.overhead


class ImageBuffers:
    def __init__(self, max_capacity):
        self.circular_buffer: deque[ImageContainer] = deque(maxlen=max_capacity)
        for i in range(0, max_capacity):
            self.circular_buffer.append(ImageContainer())

    def __len__(self):
        return len(self.circular_buffer)

    def __del__(self):
        self.clean()

    def __getitem__(self, item: int) -> ImageContainer:
        """
        Access the given buffer in the given position.
        WARNING: This method WILL NOT PERFORM ANY SYNCHRONIZATION CHECK, and it assumes that the caller
        has acquired any required lock to work with the buffer.
        :param item: the position of the buffer to access
        :return: the ImageBuffer object of the given position
        """
        return self.circular_buffer[item]

    def clean(self):
        """
        Cleans the data in _all_ the buffers of this structure
        :return:
        """
        for buffer in self.circular_buffer:
            del buffer
        gc.collect()

    def frames_ready_for_aggregation(self) -> int:
        result = 0
        for buffer in self.circular_buffer:
            if buffer.is_ready_for_aggregation():
                result += 1
        return result

    def get_next_img_lock(self) -> int:
        for i in range(0, len(self.circular_buffer)):
            instance = self.circular_buffer[i]
            if instance.try_acquire_img_lock():
                return i
        return -1

    def get_next_casc_compute_lock(self) -> int:
        for i in range(0, len(self.circular_buffer)):
            instance = self.circular_buffer[i]
            if instance.try_acquire_casc_compute_lock():
                return i
        return -1

    def get_next_aggregation_lock(self) -> int:
        for i in range(0, len(self.circular_buffer)):
            instance = self.circular_buffer[i]
            if instance.try_acquire_casc_result_available_lock():
                return i
        return -1

    def reset_buffer(self, index):
        logger.debug(f"Releasing buffer # = {index}")
        self.circular_buffer[index].try_acquire_casc_result_available_lock()
        self.circular_buffer[index].release_img_lock()
