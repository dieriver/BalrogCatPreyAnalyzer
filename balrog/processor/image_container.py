import gc
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from threading import RLock

from cv2.typing import MatLike

from balrog.utils import logger
from .cascade import EventElement


@dataclass
class _CaptureImageData:
    img_data: MatLike | None = None
    timestamp: datetime | None = None

    def __repr__(self) -> str:
        return f"<img_data: {'present' if self.img_data is not None else 'empty'}, tstamp: {self.timestamp}>"


@dataclass
class _CascadeResultData:
    event_element: EventElement | None = None
    total_runtime: float | None = None
    overhead: float | None = None


class BufferState(Enum):
    WAITING_FRAME = 1
    WAITING_CASCADE = 2
    IN_CASCADE = 3
    WAITING_AGGREGATION = 4
    USED = 5


class ImageContainer:
    def __init__(self, enable_logging: bool):
        self.enable_logging = enable_logging
        self.capture_data: _CaptureImageData = _CaptureImageData()
        self.casc_result_data: _CascadeResultData = _CascadeResultData()
        self.buffer_state = BufferState.WAITING_FRAME

    def clean(self) -> None:
        """
        Cleans the data in this image container.
        """
        self.capture_data: _CaptureImageData = _CaptureImageData()
        self.casc_result_data: _CascadeResultData = _CascadeResultData()
        self.buffer_state = BufferState.WAITING_FRAME
        gc.collect()

    def __repr__(self) -> str:
        return f"<capture_data = {repr(self.capture_data)}, result_data = {repr(self.casc_result_data)}, buffer_state = {self.buffer_state}>"

    # Methods to check the state of the locks of this buffer

    def is_ready_for_frame(self) -> bool:
        return self.buffer_state == BufferState.WAITING_FRAME

    def is_ready_for_cascade(self) -> bool:
        return self.buffer_state == BufferState.WAITING_CASCADE

    def is_ready_for_aggregation(self) -> bool:
        return self.buffer_state == BufferState.WAITING_AGGREGATION

    def has_been_used(self) -> bool:
        return self.buffer_state == BufferState.USED

    # Methods used to store the data in this buffer

    def write_capture_data(self, img_data, timestamp: datetime) -> None:
        self.capture_data = _CaptureImageData(img_data, timestamp)

    def write_cascade_data(self, event_elem: EventElement, total_time: float, overhead: float) -> bool:
        """
        Writes the cascade data to the given buffer ONLY if the buffer is going through cascade
        :param event_elem:
        :param total_time:
        :param overhead:
        :return: True if the data was written, False otherwise
        """
        if self.buffer_state is BufferState.IN_CASCADE:
            self.casc_result_data = _CascadeResultData(event_elem, total_time, overhead)
            return True
        return False

    # Accessors for the data stored in this buffer

    @property
    def get_img_data(self):
        return self.capture_data.img_data

    @property
    def get_timestamp(self) -> datetime:
        return self.capture_data.timestamp

    @property
    def get_event_element(self) -> EventElement:
        return self.casc_result_data.event_element

    @property
    def get_total_runtime(self) -> float:
        return self.casc_result_data.total_runtime

    @property
    def get_overhead(self) -> float:
        return self.casc_result_data.overhead


class ImageBuffers:
    def __init__(self, max_capacity: int, enable_logging: bool):
        """
        Creates a pre-allocated circular buffer with the given maximum capacity.
        All the indexes returned by methods of this class will return an integer in
        the range [0, max_capacity)
        :param max_capacity: the maximum capacity of the circular buffer
        """
        self.enable_logging = enable_logging
        self.circular_buffer: deque[ImageContainer] = deque(maxlen=max_capacity)
        # To emulate the circular behavior, we will keep a reference _of the first and last_
        # index of the window that is in use. When computing any "next available index" we will iterate over the range:
        # [self.base_index, self.base_index + max_capacity). Of course, this range extends to
        # the outside of the max_capacity (length) of self.circular_buffer, however, to fix that
        # (and to keep the circular behavior) we will use the integers on that range _modulus_
        # max_capacity.
        self.first_empty_frame = -1
        self.first_unprocessed_cascade = -1
        self.last_non_aggregated_frame = -1
        self.indexes_lock = RLock()

        for i in range(0, max_capacity):
            self.circular_buffer.append(ImageContainer(enable_logging))

        self.frames_available_for_frame = len(self.circular_buffer)
        self.frames_available_for_cascade = 0
        self.frames_available_for_aggregation = 0

    def __len__(self):
        return len(self.circular_buffer)

    def __del__(self):
        self.clear()

    def __getitem__(self, item: int) -> ImageContainer:
        """
        Access the given buffer in the given position.
        WARNING: This method WILL NOT PERFORM ANY SYNCHRONIZATION CHECK, and it assumes that the caller
        has acquired any required lock to work with the buffer.
        :param item: the position of the buffer to access
        :return: the ImageBuffer object of the given position
        """
        return self.circular_buffer[item]

    def _log(self, message, exception: Exception | None = None):
        if exception is not None:
            logger.exception(message)
        elif self.enable_logging:
            logger.debug(message)

    def clear(self):
        """
        Cleans the data in _all_ the buffers of this structure
        :return:
        """
        self._log("Cleaning all buffers")
        with self.indexes_lock:
            for buffer in self.circular_buffer:
                buffer.clean()

            self.first_empty_frame = -1
            self.first_unprocessed_cascade = -1
            self.last_non_aggregated_frame = -1

            self.frames_available_for_frame = len(self.circular_buffer)
            self.frames_available_for_cascade = 0
            self.frames_available_for_aggregation = 0

        gc.collect()

    def frames_ready_for_cascade(self) -> int:
        with self.indexes_lock:
            return self.frames_available_for_cascade

    def frames_ready_for_aggregation(self) -> int:
        with self.indexes_lock:
            return self.frames_available_for_aggregation

    def get_next_index_for_frame(self) -> int:
        self.indexes_lock.acquire()
        # Border case: at the start, all indexes are -1
        if self.first_empty_frame < 0:
            if not self.circular_buffer[0].is_ready_for_frame():
                self.indexes_lock.release()
                return -1
            else:
                self.first_empty_frame += 1

        # We check the indexes of the first empty frame and the last aggregated
        if self.frames_available_for_frame <= 0 or self.first_empty_frame == self.last_non_aggregated_frame:
            # If they are equal, the circular buffer is full
            self.indexes_lock.release()
            return -1
        else:
            # If they are different, we assume the frame is available
            empty_frame_index = self.first_empty_frame
            return empty_frame_index

    def mark_position_ready_for_cascade(self, index: int):
        self.first_empty_frame = ((self.first_empty_frame + 1) % len(self.circular_buffer))
        self.circular_buffer[index].buffer_state = BufferState.WAITING_CASCADE
        self.frames_available_for_frame -= 1
        self.frames_available_for_cascade += 1
        self.indexes_lock.release()

    def get_next_index_for_cascade(self) -> int:
        self.indexes_lock.acquire()
        # Border case: at the start, all indexes are -1
        if self.first_unprocessed_cascade < 0:
            if not self.circular_buffer[0].is_ready_for_cascade():
                self.indexes_lock.release()
                return -1
            else:
                self.first_unprocessed_cascade += 1

        if self.frames_available_for_cascade <= 0 or self.first_unprocessed_cascade == self.first_empty_frame:
            # There are no available frames for cascade; first unprocessed cascade does not have frame
            self.indexes_lock.release()
            return -1
        else:
            cascade_index = self.first_unprocessed_cascade
            return cascade_index

    def buffer_is_going_through_cascade(self, index: int):
        self.first_unprocessed_cascade = ((self.first_unprocessed_cascade + 1) % len(self.circular_buffer))
        self.circular_buffer[index].buffer_state = BufferState.IN_CASCADE
        self.frames_available_for_cascade -= 1
        self.indexes_lock.release()

    def mark_position_ready_for_aggregation(self, index: int):
        with self.indexes_lock:
            self.circular_buffer[index].buffer_state = BufferState.WAITING_AGGREGATION
            self.frames_available_for_aggregation += 1

    def get_next_index_for_aggregation(self) -> int:
        self.indexes_lock.acquire()
        # Border case: at the start, all indexes are -1
        if self.last_non_aggregated_frame < 0:
            if not self.circular_buffer[0].is_ready_for_aggregation():
                self.indexes_lock.release()
                return -1
            else:
                self.last_non_aggregated_frame += 1

        if self.frames_available_for_aggregation <= 0 or self.last_non_aggregated_frame == self.first_unprocessed_cascade:
            # No frames are available for aggregation; the last non aggregated frame has not gone through cascade
            self.indexes_lock.release()
            return -1
        else:
            aggregate_index = self.last_non_aggregated_frame
            return aggregate_index

    def reset_buffer(self, index):
        self._log(f"Releasing buffer # {index}")
        self.last_non_aggregated_frame = ((self.last_non_aggregated_frame + 1) % len(self.circular_buffer))
        self.circular_buffer[index].clean()
        self.frames_available_for_aggregation -= 1
        self.frames_available_for_frame += 1
        self.indexes_lock.release()
