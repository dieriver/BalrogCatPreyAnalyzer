from collections import deque
from datetime import datetime
import pytz
import time
import cv2
import gc
import os
import logging

logger = logging.getLogger("cat_logger")


class Camera:
    def __init__(self, fps):
        self.framerate = fps
        if os.getenv('CAMERA_STREAM_URI') == "":
            raise Exception("Camera stream URI not set!. Please set the 'CAMERA_STREAM_URI' environment variable")
        self.streamURL = os.getenv('CAMERA_STREAM_URI')
        self.stop_flag = False

        time.sleep(2)

    def stop_thread(self):
        logger.warning("Received signal to stop camera thread")
        self.stop_flag = True

    def fill_queue(self, main_deque: deque):
        while True:
            gc.collect()
            camera = cv2.VideoCapture(self.streamURL)

            i = 0
            while camera.isOpened():
                ret, frame = camera.read()
                main_deque.append(
                    (datetime.now(pytz.timezone('Europe/Zurich')).strftime("%Y_%m_%d_%H-%M-%S.%f"), frame)
                )
                # main_deque.pop()
                i += 1
                logger.info("Quelength: " + str(len(main_deque)))
                # print("sleeping (ms) " + str(self.framerate))
                time.sleep(1 / self.framerate)
                if i >= 60:
                    logger.info("Loop ended, starting over.")
                    camera.release()
                    del camera
                    break

                if self.stop_flag:
                    logger.warning("Terminating camera thread - A")
                    break

            if self.stop_flag:
                logger.warning("Terminating camera thread - A")
                break
        return
