import abc
import os
import time
from datetime import datetime
from logging import DEBUG, INFO, WARN, ERROR
from multiprocessing import Event
from threading import Thread

import cv2
import pytz
from cv2.typing import MatLike

from balrog.config import general_config, logging_config
from balrog.processor import ImageBuffers
from balrog.processor.cv_helpers import put_text
from balrog.utils import logger, get_resource_path


class ICamera(abc.ABC):
    def __init__(self, fps: int, frame_buffers: ImageBuffers, cleanup_threshold: int):
        self.frame_rate: int = fps
        self.cleanup_threshold: int = cleanup_threshold
        self.frame_buffers: ImageBuffers = frame_buffers
        self.stop_event: Event = Event()
        self.camera_thread: Thread = Thread(target=self.fill_queue, args=(), daemon=True, name="Camera")

    def __enter__(self):
        self.camera_thread.start()

    def __exit__(self, exception_type, exception_value, traceback):
        ICamera._log(WARN, "Stopping camera threads")
        # We set the terminate flag and wait for the thread to terminate gracefully
        if not self.stop_event.is_set():
            self.stop_event.set()
        self.camera_thread.join()

    @staticmethod
    def get_instance(
            fps: int,
            frame_buffers: ImageBuffers,
            cleanup_threshold: int,
            is_debug: bool = False
    ):
        if is_debug:
            return DbgCamera(fps, frame_buffers)
        else:
            return Camera(fps, frame_buffers, cleanup_threshold)

    @staticmethod
    def _log(level: int, message: str, exception: Exception = None):
        if exception is not None:
            logger.exception(message, exc_info=exception)
        if logging_config.enable_camera_logging:
            logger.log(level, f"Camera - {message}")

    def _write_frame_to_buffer(self, frame_data: MatLike) -> bool:
        if frame_data is None:
            # Nothing to write
            ICamera._log(DEBUG, f"Trying to write a non existing frame: {frame_data}")
            return False

        # Writing the frame to the circular buffer needs to be atomic; Let's assume we get the next
        # available buffer for a frame:
        # [AGG, AGG, CASC, CASC, AVAIL, AVAIL]
        # If we get the next frame: 4 and the release the locks. The camera thread gets preempted
        # _without writing the data_. In the meantime another thread on cascade (frames 2 or 3) gets
        # an exception, and cleans the buffer leaving this state:
        # [AVAIL, AVAIL, AVAIL, AVAIL, AVAIL, AVAIL]
        # Cascade thread gets preempted, and this one returns to CPU. We still need to write to buffer
        # number 4:
        # [AVAIL, AVAIL, AVAIL, AVAIL, RDY_CASC, AVAIL]
        # Next iterations of this thread will start writing on 0, 1... until 3 (because when cleaning,
        # the indexes were reset). Then, when trying to write the next frame to buffer 4, then it
        # encounters that is busy. Due to the invariant, this thread _assumes_ all the buffers are full,
        # so it discards the recently captured frame, and all the subsequents, leading to a stall
        # This is fixed by writing the data atomically
        tstamp = datetime.now(pytz.timezone(general_config.local_timezone))
        frame = put_text(frame_data, f"Captured: {tstamp}")
        index = self.frame_buffers.write_frame_on_next_available_buffer(
            frame,
            tstamp
        )
        if index < 0:
            ICamera._log(WARN, "Could not find a buffer ready to write an image, discarding the frame")
            return False
        else:
            ICamera._log(DEBUG, f"Frame wrote to buffer # {index}")
            return True

    @abc.abstractmethod
    def fill_queue(self) -> None:
        pass


class DbgCamera(ICamera):
    """
    Debug camera class that simply feeds a single static image into the frames
    """
    def __init__(self, fps: int, frame_buffers: ImageBuffers):
        super().__init__(fps, frame_buffers, -1)

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
    def __init__(self, fps: int, frame_buffers: ImageBuffers, cleanup_threshold: int):
        super().__init__(fps, frame_buffers, cleanup_threshold)
        stream_uri = os.getenv('CAMERA_STREAM_URI')
        if stream_uri is None or stream_uri == "":
            raise Exception("Camera stream URI not set!. Please set the 'CAMERA_STREAM_URI' environment variable")
        self.stream_url: str = stream_uri

    def fill_queue(self) -> None:
        camera = cv2.VideoCapture()
        while True:
            try:
                camera.open(self.stream_url)
                camera_fps = camera.get(cv2.CAP_PROP_FPS)
                ICamera._log(INFO, f"Capture backend name: {camera.getBackendName()}")
                ICamera._log(INFO, f"Capture FPS: {camera_fps}")
                capture_tries = 0
                previous_capture = time.time()

                while camera.isOpened():
                    # General strategy:
                    # As stated in https://stackoverflow.com/questions/52068277/change-frame-rate-in-opencv-3-4-2
                    # We cannot "time.sleep" to wait before capturing the next frame. This will end up in frames
                    # overflowing the buffer of the underlying capture backend (FFMPEG in Linux), and generating
                    # delay on the captured frames. Instead, we try the read the next available frame, and determine
                    # if it has passed enough time to match the configured fps (not the reported camera fps!)
                    success, frame = camera.read()

                    now = time.time()
                    time_elapsed = now - previous_capture
                    if time_elapsed <= 1 / self.frame_rate:
                        # The next frame won't be available until, at least, 1/camera_fps seconds
                        # It is safe to sleep until then
                        time.sleep(1 / camera_fps)
                        continue

                    # At this time, we know that it has passed, at least, 1/frame_rate secs; we can process this frame
                    previous_capture = now
                    frame_written = super()._write_frame_to_buffer(frame)
                    capture_tries += 1
                    ICamera._log(DEBUG, f"Status - Captured: {capture_tries}, last_status: {success}")

                    if not success or not frame_written:
                        ICamera._log(DEBUG, f"Frame capture not success or not written")

                    if capture_tries >= self.cleanup_threshold:
                        raise _CleanCameraException()
                    if self.stop_event.is_set():
                        raise _StopCameraException()
            except _CleanCameraException:
                ICamera._log(INFO, "Captured max configured frames; cleaning up and restarting")
            except _StopCameraException:
                ICamera._log(WARN, "Terminating camera thread")
                return
            except Exception as e:
                ICamera._log(ERROR, f'There was an exception in the camera thread!!', e)
            finally:
                if camera is not None:
                    ICamera._log(DEBUG, f"Releasing camera object")
                    camera.release()
