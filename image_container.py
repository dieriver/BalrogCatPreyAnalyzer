from threading import BoundedSemaphore
from collections import deque


class ImageBuffers:
    def __init__(self, max_capacity):
        self.circular_buffer = deque(maxlen=max_capacity)
        for i in range(0, max_capacity):
            self.circular_buffer[i] = ImageContainer()

    def get_next_img_lock(self):
        for i in range(0, len(self.circular_buffer)):
            instance = self.circular_buffer[i]
            if instance.try_acquire_img_lock():
                return i
        return -1

    def get_next_tstamp_lock(self):
        for i in range(0, len(self.circular_buffer)):
            instance = self.circular_buffer[i]
            if instance.try_acquire_tstamp_lock():
                return i
        return -1

    def get_next_casc_lock(self):
        for i in range(0, len(self.circular_buffer)):
            instance = self.circular_buffer[i]
            if instance.try_acquire_casc_lock():
                return i
        return -1

    def write_img_to_next_buffer(self, img):
        index = self.get_next_img_lock()
        self.circular_buffer[index].write_img_data(img)

    def write_tstamp_to_next_buffer(self, tstamp):
        index = self.get_next_tstamp_lock()
        self.circular_buffer[index].write_tstamp_data(tstamp)

    def write_casc_to_next_buffer(self, casc_result):
        index = self.get_next_casc_lock()
        self.circular_buffer[index].write_casc_data(casc_result)


class ImageContainer:
    def __init__(self):
        self.frame = None
        self.tstamp = None
        self.casc_data = None
        self.img_semaphore = BoundedSemaphore(1)
        self.tstamp_semaphore = BoundedSemaphore(1)
        self.casc_result_semaphore = BoundedSemaphore(1)
        self.tstamp_semaphore.acquire()
        self.casc_result_semaphore.acquire()

    def try_acquire_img_lock(self):
        return self.img_semaphore.acquire(blocking=False)

    def try_acquire_tstamp_lock(self):
        return self.tstamp_semaphore.acquire(blocking=False)

    def try_acquire_casc_lock(self):
        return self.casc_result_semaphore.acquire(blocking=False)

    def write_img_data(self, img):
        self.frame = img

    def write_tstamp_data(self, tstamp):
        self.tstamp = tstamp

    def write_casc_data(self, casc_data):
        self.casc_data = casc_data
