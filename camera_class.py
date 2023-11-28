from datetime import datetime
from threading import Thread

import pytz
import time
import cv2
import gc
import os
from utils import logger


class Camera:
    def __init__(self, fps: int, cleanup_threshold: int, frame_buffers: ImageBuffers):
        self.frame_rate = fps
        self.cleanup_threshold = cleanup_threshold
        if os.getenv('CAMERA_STREAM_URI') == "":
            raise Exception("Camera stream URI not set!. Please set the 'CAMERA_STREAM_URI' environment variable")
        self.stream_url = os.getenv('CAMERA_STREAM_URI')
        self.stop_flag = False
        self.frame_buffers = frame_buffers

        self.camera_thread = Thread(target=self.fill_queue, args=(), daemon=True)

    def __enter__(self):
        self.camera_thread.start()

    def __exit__(self, exception_type, exception_value, traceback):
        logger.warning("Stopping camera thread")
        # We set the terminate flag and wait for the thread to terminate gracefully
        self.stop_flag = True
        self.camera_thread.join()

    def fill_queue(self):
        while True:
            gc.collect()
            camera = cv2.VideoCapture(self.stream_url)

            i = 0
            while camera.isOpened():
                ret, frame = camera.read()
                self.frame_buffers.write_img_to_next_buffer(frame, datetime.now(pytz.timezone('Europe/Zurich')))
                i += 1
                logger.debug(f'Queue length: {len(self.frame_buffers)}')
                # print(f"Camera thread sleeping '{self.frame_rate}' (ms), before obtain the next frame (keep fps)")
                time.sleep(1 / self.frame_rate)
                if i >= self.cleanup_threshold:
                    logger.info("Camera captures max configured frames; cleaning up and restarting")
                    camera.release()
                    del camera
                    break

                if self.stop_flag:
                    logger.warning("Terminating camera thread - break A")
                    break

            if self.stop_flag:
                logger.warning("Terminating camera thread - B")
                break
        return
