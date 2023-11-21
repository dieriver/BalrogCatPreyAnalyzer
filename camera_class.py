from datetime import datetime
from threading import Event
import pytz
import time
import cv2
import gc
import os
from utils import logger


class Camera:
    def __init__(self, fps: int, cleanup_threshold: int, cam_rdy: Event):
        self.framerate = fps
        self.cleanup_threshold = cleanup_threshold
        self.camera_ready = cam_rdy
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
            self.camera_ready.set()
            while camera.isOpened():
                ret, frame = camera.read()
                main_deque.write_img_to_next_buffer(
                    (datetime.now(pytz.timezone('Europe/Zurich')).strftime("%Y_%m_%d_%H-%M-%S.%f"), frame)
                )
                i += 1
                logger.debug(f'Quelength: {len(main_deque)}')
                # print("sleeping (ms) " + str(self.framerate))
                time.sleep(1 / self.framerate)
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
