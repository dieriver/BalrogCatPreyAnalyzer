import gc
import os
import time
from datetime import datetime
from multiprocessing import Event
from threading import Thread

import cv2
import pytz

from balrog.processor import ImageBuffers
from balrog.utils import logger


class Camera:
    def __init__(self, fps: int, cleanup_threshold: int, frame_buffers: ImageBuffers, stop_event: Event):
        self.frame_rate = fps
        self.cleanup_threshold = cleanup_threshold
        if os.getenv('CAMERA_STREAM_URI') == "":
            raise Exception("Camera stream URI not set!. Please set the 'CAMERA_STREAM_URI' environment variable")
        self.stream_url = os.getenv('CAMERA_STREAM_URI')
        self.stop_event = stop_event
        self.frame_buffers = frame_buffers

        self.camera_thread = Thread(target=self.fill_queue, args=(), daemon=True)

    def __enter__(self):
        self.camera_thread.start()

    def __exit__(self, exception_type, exception_value, traceback):
        logger.warning("Stopping camera thread")
        # We set the terminate flag and wait for the thread to terminate gracefully
        if not self.stop_event.is_set():
            self.stop_event.set()
        self.camera_thread.join()

    def fill_queue(self):
        while True:
            gc.collect()
            camera = cv2.VideoCapture(self.stream_url)

            i = 0
            while camera.isOpened():
                ret, frame = camera.read()

                index = self.frame_buffers.get_next_index_for_frame()
                if index < 0:
                    logger.warning("Could not find a buffer ready to write an image, discarding the frame")
                    continue

                logger.debug(f"Writing frame to buffer # {index}")
                next_buffer = self.frame_buffers[index]
                next_buffer.write_capture_data(frame, datetime.now(pytz.timezone('Europe/Zurich')))
                self.frame_buffers.mark_position_ready_for_cascade(index)

                i += 1
                time.sleep(1 / self.frame_rate)
                if i >= self.cleanup_threshold:
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
