import os
import gc
import time
import cv2
import pytz
from datetime import datetime
from threading import Thread
from multiprocessing.pool import ThreadPool

from cascade import Cascade, EventElement
from detection_callbacks import send_cat_detected_message, send_dk_message, send_prey_message, send_no_prey_message
from camera_class import Camera
from telegram_bot import NodeBot
from utils import logger, cat_cam_py
from config import general_config, model_config
from image_container import ImageBuffers


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
        self.base_cascade = Cascade()
        self.EVENT_FLAG = False
        self.PATIENCE_FLAG = False
        self.CAT_DETECTED_FLAG = False
        self.FACE_FOUND_FLAG = False
        self.PREY_FLAG = None
        self.NO_PREY_FLAG = None
        self.event_objects = []
        self.patience_counter = 0
        self.event_reset_counter = 0
        self.cumulus_points = 0
        self.cat_counter = 0
        self.face_counter = 0
        self.bot = NodeBot()
        self.main_deque = ImageBuffers(2 * general_config.queue_max_threshold)
        self.camera = Camera(
            fps=general_config.camera_fps,
            cleanup_threshold=general_config.camera_cleanup_frames_threshold
        )
        self.verdict_sender_pool = ThreadPool(processes=general_config.max_message_sender_threads)
        self.camera_thread = Thread(target=self.camera.fill_queue, args=(self.main_deque,), daemon=True)

    def reset_cumuli_et_al(self):
        self.EVENT_FLAG = False
        self.PATIENCE_FLAG = False
        self.CAT_DETECTED_FLAG = False
        self.FACE_FOUND_FLAG = False
        self.PREY_FLAG = None
        self.NO_PREY_FLAG = None
        self.patience_counter = 0
        self.event_reset_counter = 0
        self.cumulus_points = 0
        self.cat_counter = 0
        self.face_counter = 0

        # Close the node_letin flag
        # TODO - This should not be modified here!
        self.bot.node_let_in_flag = False

        # for item in self.event_objects:
        #    del item
        self.event_objects.clear()

        self.main_deque.clean()

        gc.collect()

    def queue_handler(self):
        # Do this to force run all networks s.t. the network inference time stabilizes
        self.single_debug()
        self.camera_thread.start()

        while True:
            if len(self.main_deque) > general_config.queue_max_threshold:
                self.reset_cumuli_et_al()
                logger.info('EMPTYING QUEUE BECAUSE MAXIMUM THRESHOLD REACHED!')
                self.bot.send_text('Queue overflowed... emptying Queue!')
                continue

            # We check if there are enough frames to work with
            elif self.main_deque.frames_ready_for_aggregation() > general_config.min_frames_threshold:
                self.queue_worker()

            # We simply wait for new frames to be ready (The camera thread should propulate the deque)
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
        self.verdict_sender_pool.terminate()

    def queue_worker(self):
        logger.info(f'Working the Queue with len: {len(self.main_deque)}')
        # Feed the latest image in the Queue through the cascade
        next_frame_index = self.main_deque.get_next_casc_compute_lock()
        next_frame = self.main_deque[next_frame_index]

        if next_frame_index < 0:
            # We couldn't acquire the lock of a frame to compute the cascade; pass
            return

        total_runtime, cascade_obj = self.feed_to_cascade(
            target_img=next_frame.get_img_data(),
            img_name=next_frame.get_timestamp()
        )
        done_timestamp = datetime.now(pytz.timezone('Europe/Zurich'))
        logger.debug(f'Timestamp at Done Runtime: {done_timestamp}')

        overhead = (done_timestamp - next_frame.get_timestamp())
        logger.debug(f'Overhead: {overhead.total_seconds()}')

        # Add this such that the bot has some info
        self.bot.node_queue_info = len(self.main_deque)
        self.bot.node_live_img = next_frame.get_img_data()
        self.bot.node_over_head_info = overhead.total_seconds()

        # Always delete the left part
        # for i in range(self.fps_offset + 1):
        #     self.main_deque.popleft()

        if cascade_obj.cc_cat_bool:
            # We are inside an event => add event_obj to list
            logger.info('**** CAT FOUND! ****')
            self.EVENT_FLAG = True
            self.event_objects.append(cascade_obj)
            # Send a message on Telegram to ask what to do
            self.cat_counter += 1
            if self.cat_counter >= model_config.cat_counter_threshold and not self.CAT_DETECTED_FLAG:
                self.CAT_DETECTED_FLAG = True
                node_live_img_cpy = self.bot.node_live_img
                self.verdict_sender_pool.apply_async(send_cat_detected_message, args=(self.bot, node_live_img_cpy, 0,))

            # Last cat pic for bot
            self.bot.node_last_casc_img = cascade_obj.output_img

            # self.fps_offset = 0
            # If face found add the cumulus points
            if cascade_obj.face_bool:
                logger.info('**** FACE FOUND! ****')
                self.face_counter += 1
                self.cumulus_points += (50 - int(round(100 * cascade_obj.pc_prey_val)))
                self.FACE_FOUND_FLAG = True

            logger.debug(f'CUMULUS: {self.cumulus_points}')

            # Check the cumuli points and set flags if necessary
            if self.face_counter > 0 and self.PATIENCE_FLAG:
                if self.cumulus_points / self.face_counter > model_config.cumulus_no_prey_threshold:
                    self.NO_PREY_FLAG = True
                    logger.info('**** NO PREY DETECTED... YOU CLEAN... ****')
                    events_cpy = self.event_objects.copy()
                    cumuli_cpy = self.cumulus_points / self.face_counter
                    self.verdict_sender_pool.apply_async(
                        send_no_prey_message,
                        args=(self.bot, events_cpy, cumuli_cpy,)
                    )
                    self.reset_cumuli_et_al()
                elif self.cumulus_points / self.face_counter < model_config.cumulus_prey_threshold:
                    self.PREY_FLAG = True
                    logger.info('**** IT IS A PREY!!!!! ****')
                    events_cpy = self.event_objects.copy()
                    cumuli_cpy = self.cumulus_points / self.face_counter
                    self.verdict_sender_pool.apply_async(
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
            if self.event_reset_counter >= model_config.event_reset_threshold:
                # If was True => event now over => clear queue
                if self.EVENT_FLAG:
                    logger.debug('---- CLEARED QUEUE BECAUSE EVENT OVER WITHOUT CONCLUSION... ----')
                    # TODO QUICK FIX
                    if self.face_counter == 0:
                        self.face_counter = 1
                    events_cpy = self.event_objects.copy()
                    cumuli_cpy = self.cumulus_points / self.face_counter
                    self.verdict_sender_pool.apply_async(
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
