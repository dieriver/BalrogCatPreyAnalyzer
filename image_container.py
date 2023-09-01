from threading import BoundedSemaphore
from collections import deque
import logging

logger = logging.getLogger("cat_logger")


class ImageBuffers:
    def __init__(self, max_capacity):
        self.circular_buffer = deque(maxlen=max_capacity)
        for i in range(0, max_capacity):
            self.circular_buffer.append(ImageContainer())

    def __len__(self):
        return len(self.circular_buffer)

    def get_next_img_lock(self):
        for i in range(0, len(self.circular_buffer)):
            instance = self.circular_buffer[i]
            if instance.try_acquire_img_lock():
                return i
        return -1

    def get_next_casc_compute_lock(self):
        for i in range(0, len(self.circular_buffer)):
            instance = self.circular_buffer[i]
            if instance.try_acquire_casc_compute_lock():
                return i
        return -1

    # Returns the index, but it might not be useful
    def write_img_to_next_buffer(self, img):
        index = self.get_next_img_lock()
        if index < 0:
            logger.warning("Could not find a buffer ready to write an image, discarding the frame")
            return index
        logger.debug("Writing image to buffer # = " + str(index))
        self.circular_buffer[index].write_img_data(img)
        self.circular_buffer[index].release_casc_compute_lock()
        return index

    def get_image_data_from_buffer(self, index):
        return self.circular_buffer[index].get_img_data()

    def write_casc_result_to_buffer(self, index, casc_data):
        logger.debug("Writing cascade result of buffer # = " + str(index))
        self.circular_buffer[index].write_casc_data(casc_data)
        # We expect that this acquire will allways success
        self.circular_buffer[index].try_acquire_casc_compute_lock()
        self.circular_buffer[index].release_casc_res_available_lock()

    def reset_buffer(self, index):
        logger.debug("Releasing buffer # = " + str(index))
        self.circular_buffer[index].try_acquire_casc_result_available_lock()
        self.circular_buffer[index].release_img_lock()


class ImageContainer:
    def __init__(self):
        self.img_data = None
        self.casc_data = None
        self.img_semaphore = BoundedSemaphore(1)
        self.casc_compute_semaphore = BoundedSemaphore(1)
        self.casc_result_avail_semaphore = BoundedSemaphore(1)
        self.casc_compute_semaphore.acquire()
        self.casc_result_avail_semaphore.acquire()

    def try_acquire_img_lock(self):
        return self.img_semaphore.acquire(blocking=False)

    def try_acquire_casc_compute_lock(self):
        return self.casc_compute_semaphore.acquire(blocking=False)

    def try_acquire_casc_result_available_lock(self):
        return self.casc_result_avail_semaphore.acquire(blocking=False)

    def release_img_lock(self):
        self.img_semaphore.release()

    def release_casc_compute_lock(self):
        self.casc_compute_semaphore.release()

    def release_casc_res_available_lock(self):
        self.casc_result_avail_semaphore.release()

    def write_img_data(self, img_data):
        self.img_data = img_data

    def get_img_data(self):
        return self.img_data

    def write_casc_data(self, casc_data):
        self.casc_data = casc_data

    def get_casc_data(self):
        return self.casc_data
