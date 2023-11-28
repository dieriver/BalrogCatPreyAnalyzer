import gc
from datetime import datetime
from threading import BoundedSemaphore
from collections import deque
from utils import logger


class ImageContainer:
    def __init__(self):
        self.img_data = None
        self.timestamp = None
        self.casc_data = None
        self.img_semaphore = BoundedSemaphore(1)
        self.image_ready = True
        self.casc_compute_semaphore = BoundedSemaphore(1)
        self.casc_result_avail_semaphore = BoundedSemaphore(1)
        self.casc_compute_semaphore.acquire()
        self.casc_ready = False
        self.casc_result_avail_semaphore.acquire()
        self.result_ready = False

    def __del__(self) -> None:
        """
        Cleans the data in this image container.
        """
        self.img_data = None
        self.timestamp = None
        self.casc_data = None
        self.img_semaphore = BoundedSemaphore(1)
        self.casc_compute_semaphore = BoundedSemaphore(1)
        self.casc_result_avail_semaphore = BoundedSemaphore(1)
        self.casc_compute_semaphore.acquire()
        self.casc_result_avail_semaphore.acquire()

    def is_image_ready(self) -> bool:
        return self.image_ready

    def is_casc_ready(self) -> bool:
        return self.casc_ready

    def is_result_ready(self) -> bool:
        return self.result_ready

    def try_acquire_img_lock(self) -> bool:
        return self.img_semaphore.acquire(blocking=False)

    def try_acquire_casc_compute_lock(self) -> bool:
        return self.casc_compute_semaphore.acquire(blocking=False)

    def try_acquire_casc_result_available_lock(self) -> bool:
        return self.casc_result_avail_semaphore.acquire(blocking=False)

    def release_img_lock(self) -> None:
        self.img_semaphore.release()

    def release_casc_compute_lock(self) -> None:
        self.casc_compute_semaphore.release()

    def release_casc_res_available_lock(self) -> None:
        self.casc_result_avail_semaphore.release()

    def write_img_data(self, img_data) -> None:
        self.img_data = img_data

    def write_timestamp(self, timestamp: datetime) -> None:
        self.timestamp = timestamp

    def get_img_data(self):
        return self.img_data

    def get_timestamp(self) -> datetime:
        return self.timestamp

    def write_casc_data(self, casc_data) -> None:
        self.casc_data = casc_data

    def get_casc_data(self):
        return self.casc_data


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
            if buffer.is_result_ready():
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

    # Returns the index, but it might not be useful
    def write_img_to_next_buffer(self, img, timestamp: datetime) -> int:
        index = self.get_next_img_lock()
        if index < 0:
            logger.warning("Could not find a buffer ready to write an image, discarding the frame")
            return index
        logger.debug(f"Writing image to buffer # = {index}")
        self.circular_buffer[index].write_img_data(img)
        self.circular_buffer[index].release_casc_compute_lock()
        return index

    def get_image_data_from_buffer(self, index):
        return self.circular_buffer[index].get_img_data()

    def write_casc_result_to_buffer(self, index, casc_data):
        logger.debug(f"Writing cascade result of buffer # = {index}")
        self.circular_buffer[index].write_casc_data(casc_data)
        # We expect that this acquire will allways success
        self.circular_buffer[index].try_acquire_casc_compute_lock()
        self.circular_buffer[index].release_casc_res_available_lock()

    def reset_buffer(self, index):
        logger.debug(f"Releasing buffer # = {index}")
        self.circular_buffer[index].try_acquire_casc_result_available_lock()
        self.circular_buffer[index].release_img_lock()
