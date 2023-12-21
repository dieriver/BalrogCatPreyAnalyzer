import copy
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from threading import RLock
from typing import Self

from cv2.typing import MatLike

from balrog.utils import logger
from balrog.processor import EventElement


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


class _BufferState(Enum):
    WAITING_FRAME = 1
    IN_FRAME = 2
    WAITING_CASCADE = 3
    IN_CASCADE = 4
    WAITING_AGGREGATION = 5
    IN_AGGREGATION = 6
    USED = 7


class ImageContainer:
    def __init__(
            self,
            enable_logging: bool,
            capture_data: _CaptureImageData = None,
            casc_result_data: _CascadeResultData = None,
            buffer_state: _BufferState = None
    ):
        self._enable_logging: bool = enable_logging
        self._capture_data: _CaptureImageData = _CaptureImageData() if capture_data is None else capture_data
        self._casc_result_data: _CascadeResultData = _CascadeResultData() if casc_result_data is None else casc_result_data
        self._buffer_state = _BufferState.WAITING_FRAME if buffer_state is None else buffer_state

    def __repr__(self) -> str:
        return (f"<Buff State: {self._buffer_state}>, "
                f"Capture data: {repr(self._capture_data)},>")

    def clone(self) -> Self:
        if self._capture_data.img_data is None:
            logger.debug(f"IMG DATA IS NONE")
        return ImageContainer(
            self.enable_logging,
            _CaptureImageData(self._capture_data.img_data.copy(), self._capture_data.timestamp),
            copy.deepcopy(self._casc_result_data),
            self.buffer_state
        )

    def clean(self) -> None:
        """
        Cleans the data in this image container.
        """
        self._capture_data: _CaptureImageData = _CaptureImageData()
        self._casc_result_data: _CascadeResultData = _CascadeResultData()
        self._buffer_state = _BufferState.WAITING_FRAME

    @property
    def enable_logging(self) -> bool:
        return self._enable_logging

    @property
    def buffer_state(self) -> _BufferState:
        return self._buffer_state

    @buffer_state.setter
    def buffer_state(self, new_state: _BufferState) -> None:
        self._buffer_state = new_state

    @property
    def capture_data(self) -> _CaptureImageData:
        return self._capture_data

    @capture_data.setter
    def capture_data(self, data: _CaptureImageData) -> None:
        self._capture_data = data

    @property
    def casc_result_data(self) -> _CascadeResultData:
        return self._casc_result_data

    @casc_result_data.setter
    def casc_result_data(self, data: _CascadeResultData) -> None:
        self._casc_result_data = data

    # Properties used to check the state of the locks of this buffer
    @property
    def is_ready_for_frame(self) -> bool:
        return self.buffer_state == _BufferState.WAITING_FRAME

    @property
    def is_ready_for_cascade(self) -> bool:
        return self.buffer_state == _BufferState.WAITING_CASCADE

    @property
    def is_ready_for_aggregation(self) -> bool:
        return self.buffer_state == _BufferState.WAITING_AGGREGATION

    @property
    def has_been_used(self) -> bool:
        return self.buffer_state == _BufferState.USED

    # Methods used to store the data in this buffer
    def write_capture_data(self, img_data: MatLike, timestamp: datetime) -> None:
        self._capture_data = _CaptureImageData(img_data.copy(), timestamp)

    # Accessors for the data stored in this buffer
    @property
    def img_data(self):
        return self.capture_data.img_data

    @property
    def timestamp(self) -> datetime:
        return self.capture_data.timestamp

    @property
    def event_element(self) -> EventElement:
        return self.casc_result_data.event_element

    @property
    def total_runtime(self) -> float:
        return self.casc_result_data.total_runtime

    @property
    def overhead(self) -> float:
        return self.casc_result_data.overhead


class ImageBuffers:
    def __init__(self, max_capacity: int, enable_logging: bool):
        """
        Creates a pre-allocated circular buffer with the given maximum capacity.
        All the indexes returned by methods of this class will return an integer in
        the range [0, max_capacity)
        :param max_capacity: the maximum capacity of the circular buffer
        """
        self._enable_logging = enable_logging
        self._circular_buffer: deque[ImageContainer] = deque(maxlen=max_capacity)
        # To emulate the circular behavior, we will keep a reference _of the first and last_
        # index of the window that is in use. When computing any "next available index" we will iterate over the range:
        # [self.base_index, self.base_index + max_capacity). Of course, this range extends to
        # the outside of the max_capacity (length) of self.circular_buffer, however, to fix that
        # (and to keep the circular behavior) we will use the integers on that range _modulus_
        # max_capacity.
        self._first_empty_frame = -1
        self._first_unprocessed_cascade = -1
        self._last_non_aggregated_frame = -1
        self._indexes_lock = RLock()

        for i in range(0, max_capacity):
            self._circular_buffer.append(ImageContainer(enable_logging))

        self._frames_available_for_frame = len(self._circular_buffer)
        self._frames_available_for_cascade = 0
        self._frames_available_for_aggregation = 0

    def __len__(self):
        return len(self._circular_buffer)

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
        return self._circular_buffer[item]

    def _log(self, message, exception: Exception | None = None) -> None:
        if exception is not None:
            logger.exception(message)
        elif self._enable_logging:
            logger.debug(message)

    def clear(self) -> None:
        """
        Cleans the data in _all_ the buffers of this structure
        :return:
        """
        self._log("Cleaning all buffers")
        with self._indexes_lock:
            for buffer in self._circular_buffer:
                buffer.clean()

            self._first_empty_frame = -1
            self._first_unprocessed_cascade = -1
            self._last_non_aggregated_frame = -1

            self._frames_available_for_frame = len(self._circular_buffer)
            self._frames_available_for_cascade = 0
            self._frames_available_for_aggregation = 0

    def frames_ready_for_cascade(self) -> int:
        with self._indexes_lock:
            return self._frames_available_for_cascade

    def frames_ready_for_aggregation(self) -> int:
        with self._indexes_lock:
            return self._frames_available_for_aggregation

    def get_next_index_for_frame(self) -> int:
        with self._indexes_lock:
            # Border case: at the start, all indexes are -1
            if self._first_empty_frame < 0:
                if not self._circular_buffer[0].is_ready_for_frame:
                    return -1
                else:
                    self._first_empty_frame += 1

            # We check the state of the potential next buffer
            buffer_state = self._circular_buffer[self._first_empty_frame].buffer_state
            if self._frames_available_for_frame <= 0 or buffer_state != _BufferState.WAITING_FRAME:
                # If they are equal, the circular buffer is full
                return -1
            else:
                # If they are different, we assume the frame is available
                empty_frame_index = self._first_empty_frame
                self._circular_buffer[empty_frame_index].buffer_state = _BufferState.IN_FRAME
                self._first_empty_frame = ((self._first_empty_frame + 1) % len(self._circular_buffer))
                self._frames_available_for_frame -= 1
                return empty_frame_index

    def mark_position_ready_for_cascade(self, index: int) -> None:
        with self._indexes_lock:
            self._circular_buffer[index].buffer_state = _BufferState.WAITING_CASCADE
            self._frames_available_for_cascade += 1

    def get_next_index_for_cascade(self) -> int:
        with self._indexes_lock:
            # Border case: at the start, all indexes are -1
            if self._first_unprocessed_cascade < 0:
                if not self._circular_buffer[0].is_ready_for_cascade:
                    return -1
                else:
                    self._first_unprocessed_cascade += 1

            buffer_state = self._circular_buffer[self._first_unprocessed_cascade].buffer_state
            if self._frames_available_for_cascade <= 0 or buffer_state != _BufferState.WAITING_CASCADE:
                # There are no available frames for cascade; first unprocessed cascade does not have frame
                return -1
            else:
                cascade_index = self._first_unprocessed_cascade
                self._circular_buffer[cascade_index].buffer_state = _BufferState.IN_CASCADE
                self._first_unprocessed_cascade = ((self._first_unprocessed_cascade + 1) % len(self._circular_buffer))
                self._frames_available_for_cascade -= 1
                return cascade_index

    def write_cascade_data(self, index: int, event_elem: EventElement, total_time: float, overhead: float) -> None:
        with self._indexes_lock:
            if self._circular_buffer[index].buffer_state == _BufferState.IN_CASCADE:
                self._circular_buffer[index].casc_result_data = _CascadeResultData(event_elem, total_time, overhead)
                self._circular_buffer[index].buffer_state = _BufferState.WAITING_AGGREGATION
                self._frames_available_for_aggregation += 1

    def get_next_index_for_aggregation(self) -> int:
        with self._indexes_lock:
            # Border case: at the start, all indexes are -1
            if self._last_non_aggregated_frame < 0:
                if not self._circular_buffer[0].is_ready_for_aggregation:
                    return -1
                else:
                    self._last_non_aggregated_frame += 1

            buffer_state = self._circular_buffer[self._last_non_aggregated_frame].buffer_state
            if self._frames_available_for_aggregation <= 0 or buffer_state != _BufferState.WAITING_AGGREGATION:
                # No frames are available for aggregation; the last non aggregated frame has not gone through cascade
                return -1
            else:
                aggregate_index = self._last_non_aggregated_frame
                self._last_non_aggregated_frame = ((self._last_non_aggregated_frame + 1) % len(self._circular_buffer))
                self._frames_available_for_aggregation -= 1
                return aggregate_index

    def reset_buffer(self, index: int) -> None:
        with self._indexes_lock:
            self._log(f"Releasing buffer # {index}")
            self._circular_buffer[index].clean()
            self._frames_available_for_frame += 1
