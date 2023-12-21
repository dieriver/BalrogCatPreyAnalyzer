import abc
import gc
import os
import time
from datetime import datetime
from multiprocessing import Event
from threading import Thread
from typing import Self

import cv2
import pytz
from cv2.typing import MatLike

from balrog.processor import ImageBuffers
from balrog.utils import logger, get_resource_path


class ICamera(abc.ABC):
    def __init__(self, fps: int, frame_buffers: ImageBuffers, stop_event: Event, cleanup_threshold: int):
        self.frame_rate = fps
        self.cleanup_threshold = cleanup_threshold
        self.frame_buffers = frame_buffers
        self.stop_event = stop_event
        self.camera_thread = Thread(target=self.fill_queue, args=(), daemon=True)

    def __enter__(self):
        self.camera_thread.start()

    def __exit__(self, exception_type, exception_value, traceback):
        logger.warning("Stopping camera thread")
        # We set the terminate flag and wait for the thread to terminate gracefully
        if not self.stop_event.is_set():
            self.stop_event.set()
        self.camera_thread.join()

    @classmethod
    def get_instance(
            cls,
            fps: int,
            frame_buffers: ImageBuffers,
            stop_event: Event,
            cleanup_threshold: int,
            is_debug: bool = False
    ) -> Self:
        if is_debug:
            return DbgCamera(fps, frame_buffers, stop_event)
        else:
            return Camera(fps, frame_buffers, stop_event, cleanup_threshold)

    def _write_frame_to_buffer(self, frame_data: MatLike) -> bool:
        index = self.frame_buffers.get_next_index_for_frame()
        if index < 0:
            logger.warning("Could not find a buffer ready to write an image, discarding the frame")
            return False

        logger.debug(f"Writing frame to buffer # {index}")
        next_buffer = self.frame_buffers[index]
        next_buffer.write_capture_data(frame_data, datetime.now(pytz.timezone('Europe/Amsterdam')))
        self.frame_buffers.mark_position_ready_for_cascade(index)
        return True

    @abc.abstractmethod
    def fill_queue(self) -> None:
        pass


class DbgCamera(ICamera):
    """
    Debug camera class that simply feeds a single static image into the frames
    """
    def __init__(self, fps: int, frame_buffers: ImageBuffers, stop_event: Event):
        super().__init__(fps, frame_buffers, stop_event, -1)

    def fill_queue(self) -> None:
        with get_resource_path("dbg_casc.jpg") as resource:
            frame = cv2.imread(str(resource))
        while True:
            super()._write_frame_to_buffer(frame)
            time.sleep(1 / self.frame_rate)

            if self.stop_event.is_set():
                logger.warning("Terminating debug camera thread")
                return


class Camera(ICamera):
    def __init__(self, fps: int, frame_buffers: ImageBuffers, stop_event: Event, cleanup_threshold: int):
        super().__init__(fps, frame_buffers, stop_event, cleanup_threshold)
        self.frame_rate = fps
        self.cleanup_threshold = cleanup_threshold
        if os.getenv('CAMERA_STREAM_URI') == "":
            raise Exception("Camera stream URI not set!. Please set the 'CAMERA_STREAM_URI' environment variable")
        self.stream_url = os.getenv('CAMERA_STREAM_URI')
        self.stop_event = stop_event
        self.frame_buffers = frame_buffers

    def fill_queue(self) -> None:
        while True:
            camera = cv2.VideoCapture(self.stream_url)

            i = 0
            while camera.isOpened():
                success, frame = camera.read()
                if not success or not super()._write_frame_to_buffer(frame):
                    # Frame capture was not successful or it could not be written to the buffer
                    continue

                i += 1
                time.sleep(1 / self.frame_rate)
                if 0 < self.cleanup_threshold <= i:
                    logger.info("Camera captures max configured frames; cleaning up and restarting")
                    camera.release()
                    del camera
                    break

                if self.stop_event.is_set():
                    logger.warning("Terminating camera thread - break A")
                    break

            if self.stop_event.is_set():
                logger.warning("Terminating camera thread - break B")
                return
