from datetime import datetime
import pytz
import time
import cv2
import gc
import os
from utils import logger


class Camera:
    def __init__(self, fps: int, cleanup_threshold: int):
        self.frame_rate = fps
        self.cleanup_threshold = cleanup_threshold
        if os.getenv('CAMERA_STREAM_URI') == "":
            raise Exception("Camera stream URI not set!. Please set the 'CAMERA_STREAM_URI' environment variable")
        self.streamURL = os.getenv('CAMERA_STREAM_URI')
        self.stop_flag = False

        time.sleep(2)

    def stop_thread(self):
        logger.warning("Received signal to stop camera thread")
        self.stop_flag = True

    def fill_queue(self, main_deque: ImageBuffers):
        while True:
            gc.collect()
            camera = cv2.VideoCapture(self.streamURL)

            i = 0
            while camera.isOpened():
                ret, frame = camera.read()
                main_deque.write_img_to_next_buffer(frame, datetime.now(pytz.timezone('Europe/Zurich')))
                i += 1
                logger.debug(f'Queue length: {len(main_deque)}')
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
