import os
import gc
import time
import cv2
import pytz
from datetime import datetime
from collections import deque
from threading import Thread
from multiprocessing.pool import ThreadPool
import xml.etree.ElementTree as ET

from cascade import Cascade, EventElement
from detection_callbacks import send_cat_detected_message, send_dk_message, send_prey_message, send_no_prey_message
from camera_class import Camera
from telegram_bot import NodeBot
from utils import logger, cat_cam_py


class SequentialCascadeFeeder:
    """
    Implementation of the main loop of the software. This class:
      * Starts the thread that reads the frames from the camera class and puts them on a queue
      * Starts the main loop which:
      * Constantly checks the queue from the camera
      * Reads a frame from the queue (if there are enough frames)
      * Invokes the cascade on the frame to compute the results
      * Process the results of the cascade, computing cumulative with previous frames' results
      * Invokes the telegram callbacks with the verdicts.
    TODO - The main loop of this class (method `queue_worker`) takes a lot of responsibility.
    It would be great to refactor that code
    """
    def __init__(self):
        self.log_dir = os.path.join(os.getcwd(), 'log')
        logger.info(f'Log Dir: {self.log_dir}')
        self.event_nr = 0
        self.base_cascade = Cascade()
        self.DEFAULT_FPS_OFFSET = 3
        self.QUEUE_MAX_THRESHOLD = 100
        self.fps_offset = self.DEFAULT_FPS_OFFSET
        self.MAX_PROCESSES = 5
        self.EVENT_FLAG = False
        self.event_objects = []
        self.patience_counter = 0
        self.PATIENCE_FLAG = False
        self.CAT_DETECTED_FLAG = False
        self.FACE_FOUND_FLAG = False
        self.event_reset_threshold = 6
        self.event_reset_counter = 0
        self.cat_counter = 0
        self.cat_counter_threshold = 6
        self.cumulus_points = 0
        self.cumulus_prey_threshold = -10
        self.cumulus_no_prey_threshold = 2.9603
        self.prey_val_hard_threshold = 0.6
        self.face_counter = 0
        self.PREY_FLAG = None
        self.NO_PREY_FLAG = None
        self.bot = NodeBot()
        self.main_deque = deque()
        self.camera = Camera(fps=10)
        self.processing_pool = ThreadPool(processes=4)
        self.camera_thread = Thread(target=self.camera.fill_queue, args=(self.main_deque,), daemon=True)

    def reset_cumuli_et_al(self):
        self.EVENT_FLAG = False
        self.patience_counter = 0
        self.PATIENCE_FLAG = False
        self.CAT_DETECTED_FLAG = False
        self.FACE_FOUND_FLAG = False
        self.cumulus_points = 0
        self.fps_offset = self.DEFAULT_FPS_OFFSET
        self.event_reset_counter = 0
        self.cat_counter = 0
        self.face_counter = 0
        self.PREY_FLAG = None
        self.NO_PREY_FLAG = None
        self.cumulus_points = 0

        # Close the node_letin flag
        self.bot.node_let_in_flag = False

        #for item in self.event_objects:
        #    del item
        self.event_objects.clear()

        for item in self.main_deque:
            del item
        self.main_deque.clear()

        gc.collect()

    def queue_handler(self):
        # Do this to force run all networks s.t. the network inference time stabilizes
        self.single_debug()
        self.camera_thread.start()

        while True:
            if len(self.main_deque) > self.QUEUE_MAX_THRESHOLD:
                self.reset_cumuli_et_al()
                logger.info('EMPTYING QUEUE BECAUSE MAXIMUM THRESHOLD REACHED!')
                self.bot.send_text('Queue overflowed... emptying Queue!')

            elif len(self.main_deque) > self.DEFAULT_FPS_OFFSET:
                self.queue_worker()

            else:
                logger.info(f'Nothing to work with => Queue length: {len(self.main_deque)}')
                time.sleep(0.25)

            # Check if user force opens the door
            if self.bot.node_let_in_flag:
                # We do super simple stuff here. The actual unlock of the door is handled in NodeBot class
                self.reset_cumuli_et_al()

    def shutdown(self):
        # We stop the camera thread and the thread pool
        self.camera.stop_thread()
        self.camera_thread.join()
        self.processing_pool.terminate()

    def queue_worker(self):
        logger.info(f'Working the Queue with len: {len(self.main_deque)}')
        start_time = time.time()
        # Feed the latest image in the Queue through the cascade
        total_runtime, cascade_obj = self.feed_to_cascade(
            target_img=self.main_deque[self.fps_offset][1],
            img_name=self.main_deque[self.fps_offset][0]
        )
        current_time = time.time()
        logger.debug(f'Runtime: {current_time - start_time}')
        done_timestamp = datetime.now(pytz.timezone('Europe/Zurich')).strftime("%Y_%m_%d_%H-%M-%S.%f")
        logger.debug(f'Timestamp at Done Runtime: {done_timestamp}')

        overhead = (datetime.strptime(done_timestamp, "%Y_%m_%d_%H-%M-%S.%f") -
                    datetime.strptime(self.main_deque[self.fps_offset][0], "%Y_%m_%d_%H-%M-%S.%f")
                    )
        logger.debug('Overhead:' + str(overhead.total_seconds()))

        # Add this such that the bot has some info
        self.bot.node_queue_info = len(self.main_deque)
        self.bot.node_live_img = self.main_deque[self.fps_offset][1]
        self.bot.node_over_head_info = overhead.total_seconds()

        # Always delete the left part
        for i in range(self.fps_offset + 1):
            self.main_deque.popleft()

        if cascade_obj.cc_cat_bool:
            # We are inside an event => add event_obj to list
            logger.info('**** CAT FOUND! ****')
            self.EVENT_FLAG = True
            self.event_nr = self.get_event_nr()
            self.event_objects.append(cascade_obj)
            # Send a message on Telegram to ask what to do
            self.cat_counter += 1
            if self.cat_counter >= self.cat_counter_threshold and not self.CAT_DETECTED_FLAG:
                self.CAT_DETECTED_FLAG = True
                node_live_img_cpy = self.bot.node_live_img
                self.processing_pool.apply_async(send_cat_detected_message, args=(self.bot, node_live_img_cpy, 0,))

            # Last cat pic for bot
            self.bot.node_last_casc_img = cascade_obj.output_img

            self.fps_offset = 0
            # If face found add the cumulus points
            if cascade_obj.face_bool:
                logger.info('**** FACE FOUND! ****')
                self.face_counter += 1
                self.cumulus_points += (50 - int(round(100 * cascade_obj.pc_prey_val)))
                self.FACE_FOUND_FLAG = True

            logger.debug(f'CUMULUS: {self.cumulus_points}')

            # Check the cumuli points and set flags if necessary
            if self.face_counter > 0 and self.PATIENCE_FLAG:
                if self.cumulus_points / self.face_counter > self.cumulus_no_prey_threshold:
                    self.NO_PREY_FLAG = True
                    logger.info('**** NO PREY DETECTED... YOU CLEAN... ****')
                    events_cpy = self.event_objects.copy()
                    cumuli_cpy = self.cumulus_points / self.face_counter
                    self.processing_pool.apply_async(
                        send_no_prey_message,
                        args=(self.bot, events_cpy, cumuli_cpy,)
                    )
                    self.reset_cumuli_et_al()
                elif self.cumulus_points / self.face_counter < self.cumulus_prey_threshold:
                    self.PREY_FLAG = True
                    logger.info('**** IT IS A PREY!!!!! ****')
                    events_cpy = self.event_objects.copy()
                    cumuli_cpy = self.cumulus_points / self.face_counter
                    self.processing_pool.apply_async(
                        send_prey_message,
                        args=(self.bot, events_cpy, cumuli_cpy,)
                    )
                    self.reset_cumuli_et_al()
                else:
                    self.NO_PREY_FLAG = False
                    self.PREY_FLAG = False

            # Cat was found => still belongs to event => acts as dk state
            self.event_reset_counter = 0
            self.cat_counter = 0

        # No cat detected => reset event_counters if necessary
        else:
            logger.info('**** NO CAT FOUND! ****')
            self.event_reset_counter += 1
            if self.event_reset_counter >= self.event_reset_threshold:
                # If was True => event now over => clear queue
                if self.EVENT_FLAG:
                    logger.debug('---- CLEARED QUEUE BECAUSE EVENT OVER WITHOUT CONCLUSION... ----')
                    # TODO QUICK FIX
                    if self.face_counter == 0:
                        self.face_counter = 1
                    events_cpy = self.event_objects.copy()
                    cumuli_cpy = self.cumulus_points / self.face_counter
                    self.processing_pool.apply_async(
                        send_dk_message,
                        args=(self.bot, events_cpy, cumuli_cpy,)
                    )
                self.reset_cumuli_et_al()

        if self.EVENT_FLAG and self.FACE_FOUND_FLAG:
            self.patience_counter += 1
        if self.patience_counter > 2:
            self.PATIENCE_FLAG = True
        if self.face_counter > 1:
            self.PATIENCE_FLAG = True

    def feed_to_cascade(self, target_img, img_name):
        target_event_obj = EventElement(img_name=img_name, cc_target_img=target_img)

        start_time = time.time()
        single_cascade = self.base_cascade.do_single_cascade(event_img_object=target_event_obj)
        single_cascade.total_inference_time = sum(filter(None, [
            single_cascade.cc_inference_time,
            single_cascade.cr_inference_time,
            single_cascade.bbs_inference_time,
            single_cascade.haar_inference_time,
            single_cascade.ff_bbs_inference_time,
            single_cascade.ff_haar_inference_time,
            single_cascade.pc_inference_time]))
        total_runtime = time.time() - start_time
        logger.debug(f'Total Runtime: {total_runtime}')

        return total_runtime, single_cascade

    def single_debug(self):
        start_time = time.time()
        target_img_name = 'dummy_img.jpg'
        target_img = cv2.imread(
            os.path.join(cat_cam_py, 'readme_images/lenna_casc_Node1_001557_02_2020_05_24_09-49-35.jpg')
        )
        cascade_obj = self.feed_to_cascade(target_img=target_img, img_name=target_img_name)[1]
        current_time = time.time()
        logger.debug(f'Runtime: {current_time - start_time}')
        return cascade_obj

    def get_event_nr(self):
        tree = ET.parse(os.path.join(self.log_dir, 'info.xml'))
        data = tree.getroot()
        img_nr = int(data.find('node').get('imgNr'))
        data.find('node').set('imgNr', str(int(img_nr) + 1))
        tree.write(os.path.join(self.log_dir, 'info.xml'))
        return img_nr
