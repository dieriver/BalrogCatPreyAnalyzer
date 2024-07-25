import abc
import os
import time
from datetime import datetime
from logging import DEBUG, INFO, WARN
from multiprocessing import Event
from threading import Thread

import cv2
import pytz
from cv2.typing import MatLike

from balrog.config import general_config, logging_config
from balrog.processor import ImageBuffers
from balrog.utils import logger, get_resource_path


class ICamera(abc.ABC):
    def __init__(self, fps: int, frame_buffers: ImageBuffers, stop_event: Event, cleanup_threshold: int):
        self.frame_rate: int = fps
        self.cleanup_threshold: int = cleanup_threshold
        self.frame_buffers: ImageBuffers = frame_buffers
        self.stop_event: Event = stop_event
        self.camera_thread: Thread = Thread(target=self.fill_queue, args=(), daemon=True)

    def __enter__(self):
        self.camera_thread.start()

    def __exit__(self, exception_type, exception_value, traceback):
        ICamera._log(WARN, "Stopping camera thread")
        # We set the terminate flag and wait for the thread to terminate gracefully
        if not self.stop_event.is_set():
            self.stop_event.set()
        self.camera_thread.join()

    @staticmethod
    def get_instance(
            fps: int,
            frame_buffers: ImageBuffers,
            stop_event: Event,
            cleanup_threshold: int,
            is_debug: bool = False
    ):
        if is_debug:
            return DbgCamera(fps, frame_buffers, stop_event)
        else:
            return Camera(fps, frame_buffers, stop_event, cleanup_threshold)

    @staticmethod
    def _log(level: int, message: str):
        if logging_config.enable_camera_logging:
            logger.log(level, f"{message}")

    def _write_frame_to_buffer(self, frame_data: MatLike) -> bool:
        index = self.frame_buffers.get_next_index_for_frame()
        if index < 0:
            ICamera._log(WARN, "Could not find a buffer ready to write an image, discarding the frame")
            return False

        ICamera._log(DEBUG, f"Writing frame to buffer # {index}")
        next_buffer = self.frame_buffers[index]
        next_buffer.write_capture_data(frame_data, datetime.now(pytz.timezone(general_config.local_timezone)))
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
                ICamera._log(WARN, "Terminating debug camera thread")
                return


class _StopCameraException(Exception):
    pass


class _CleanCameraException(Exception):
    pass


class Camera(ICamera):
    def __init__(self, fps: int, frame_buffers: ImageBuffers, stop_event: Event, cleanup_threshold: int):
        super().__init__(fps, frame_buffers, stop_event, cleanup_threshold)
        stream_uri = os.getenv('CAMERA_STREAM_URI')
        if stream_uri is None or stream_uri == "":
            raise Exception("Camera stream URI not set!. Please set the 'CAMERA_STREAM_URI' environment variable")
        self.stream_url: str = stream_uri

    def fill_queue(self) -> None:
        camera = None
        while True:
            try:
                camera = cv2.VideoCapture(self.stream_url)
                captured_frames = 0
                while camera.isOpened():
                    success, frame = camera.read()
                    frame_written = super()._write_frame_to_buffer(frame)
                    ICamera._log(DEBUG, f"Status - Captured: {captured_frames}, last_status:{success}")
                    captured_frames += 1

                    time.sleep(1 / self.frame_rate)

                    if not success or not frame_written:
                        ICamera._log(DEBUG, f"Frame capture not success or not written")
                        # Frame capture was not successful, or it could not be written to the buffer
                        # try again
                        continue
                    if 0 < self.cleanup_threshold <= captured_frames:
                        raise _CleanCameraException()
                    if self.stop_event.is_set():
                        raise _StopCameraException()
            except _CleanCameraException:
                ICamera._log(INFO, "Captured max configured frames; cleaning up and restarting")
            except _StopCameraException:
                ICamera._log(WARN, "Terminating camera thread")
                return
            finally:
                if camera is not None:
                    ICamera._log(DEBUG, f"Releasing camera object")
                    camera.release()
                    del camera
