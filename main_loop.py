import gc
import os
import sys
import time
from datetime import datetime
from multiprocessing import Event
from multiprocessing.pool import ThreadPool

import cv2
import pytz

from cascade import Cascade, EventElement
from config import general_config, model_config
from detection_callbacks import send_cat_detected_message, send_dk_message, send_prey_message, send_no_prey_message
from image_container import ImageBuffers
from telegram_bot import BalrogTelegramBot, DebugBot
from utils import logger, cat_cam_py


class FrameResultAggregator:
    """
    Implementation of the aggregation loop of the software. This class:
      * Starts the thread that reads the already-processed frames from the shared circular buffer
      * Starts the aggregation process which:
      * Constantly checks the circular buffer for frames ready to be aggregated
      * Reads a frame from the buffer (if there are enough ready frames)
      * Aggregates the results, computing cumulative with previous frames' results
      * Invokes the telegram callbacks with the verdicts.
    """
    def __init__(self, frame_buffers: ImageBuffers, stop_event: Event):
        self.clean_queue_event: Event = Event()
        self.stop_event = stop_event
        if os.getenv("BALROG_USE_NULL_BOT") is not None:
            self.bot = DebugBot()
        else:
            self.bot = BalrogTelegramBot(self.clean_queue_event, stop_event)
        self.verdict_sender_pool = ThreadPool(processes=general_config.max_message_sender_threads)
        # Aggregation fields
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
        self.event_objects = []
        self.frame_buffers = frame_buffers

    def __enter__(self):
        # We don't do anything here
        pass

    def __exit__(self, exception_type, exception_value, traceback):
        self.verdict_sender_pool.terminate()
        if exception_type is not None:
            logger.error(f"Something wrong happened in the frame result aggregator thread")
            logger.error(f"Exception type: {repr(exception_type)}")
        if exception_value is not None:
            logger.error(f"Exception value: {exception_value}")
        if traceback is not None:
            logger.error(f"Traceback: {repr(traceback)}")
            sys.exit(1)
        # We use a "successful" exit code to restart the script
        # This is interpreted as a call to restart the script
        sys.exit(0)

    def reset_aggregation_fields(self):
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
        self.event_objects.clear()
        self.frame_buffers.clear()
        self.clean_queue_event.clear()

    def queue_handler(self):
        try:
            while not self.stop_event.is_set():
                # We check if there are enough frames to work with (according to the config)
                frames_rdy_for_aggregation = self.frame_buffers.frames_ready_for_aggregation()
                logger.debug(f"Frames ready for aggregation: {frames_rdy_for_aggregation}")

                if frames_rdy_for_aggregation >= general_config.min_aggregation_frames_threshold:
                    # Here we go :)
                    self.aggregate_available_frames(frames_rdy_for_aggregation)
                else:
                    # We simply wait for new frames to be ready (The camera thread should propulate the deque)
                    time.sleep(0.25)

                # Check if user force opens the door
                if self.clean_queue_event.is_set():
                    # We do super simple stuff here. The actual unlock of the door is handled in NodeBot class
                    self.reset_aggregation_fields()
        except Exception as e:
            logger.exception("Exception in aggregation thread: ", e)

    def aggregate_available_frames(self, frames_rdy_for_aggregation: int):
        # We get the last buffer, and extract its data
        next_frame_index = self.frame_buffers.get_next_index_for_aggregation()
        if next_frame_index < 0:
            return

        next_frame = self.frame_buffers[next_frame_index]
        cascade_obj = next_frame.get_event_element()
        overhead = next_frame.get_overhead()
        image_data = next_frame.get_img_data()

        # We release the lock asap
        self.frame_buffers.reset_buffer(next_frame_index)

        # Add this such that the bot has some info
        # TODO - Directly access this data is not recommended. Refactor this!
        self.bot.node_queue_info = frames_rdy_for_aggregation
        self.bot.node_live_img = image_data
        self.bot.node_over_head_info = overhead

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
                    self.reset_aggregation_fields()
                elif self.cumulus_points / self.face_counter < model_config.cumulus_prey_threshold:
                    self.PREY_FLAG = True
                    logger.info('**** IT IS A PREY!!!!! ****')
                    events_cpy = self.event_objects.copy()
                    cumuli_cpy = self.cumulus_points / self.face_counter
                    self.verdict_sender_pool.apply_async(
                        send_prey_message,
                        args=(self.bot, events_cpy, cumuli_cpy,)
                    )
                    self.reset_aggregation_fields()
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
                self.reset_aggregation_fields()

        if self.EVENT_FLAG and self.FACE_FOUND_FLAG:
            self.patience_counter += 1
        if self.patience_counter > 2:
            self.PATIENCE_FLAG = True
        if self.face_counter > 1:
            self.PATIENCE_FLAG = True


class FrameProcessor:
    """
    Implementation of the main loop of the software. This class:
      * Starts the thread that reads the frames stored by the camera thread in the shared circular buffer
      * Starts the main loop which:
      * Constantly checks the circular buffer for images ready to be processed
      * Reads a frame from the buffer (if there are enough frames)
      * Invokes the cascade on the frame to compute the results
      * Writes the results to the circular buffer
      * Marks the buffer as ready to be aggregated
    """
    def __init__(self, frame_buffers: ImageBuffers, stop_event: Event):
        self.stop_event = stop_event
        self.base_cascade = Cascade()
        self.frame_buffers = frame_buffers
        self.frame_processor_pool = ThreadPool(processes=general_config.max_message_sender_threads)

    def __enter__(self):
        # Do this to force run all networks s.t. the network inference time stabilizes
        self.single_debug()
        # We need to submit the process tasks here
        for i in range(0, general_config.max_frame_processor_threads):
            self.frame_processor_pool.apply_async(func=self.process_frame, args=(i,))

    def __exit__(self, exception_type, exception_value, traceback):
        self.frame_processor_pool.terminate()
        if exception_type is not None:
            logger.error(f"Something wrong happened in the frame processor thread")
            logger.error(f"Exception type: {exception_type}")
        if exception_value is not None:
            logger.error(f"Exception value: {exception_value}")
        if traceback is not None:
            logger.error(f"Traceback: {traceback}")
        return True

    def feed_to_cascade(self, target_img, img_name, thread_id: int) -> tuple[float, EventElement]:
        target_event_obj = EventElement(img_name=img_name, cc_target_img=target_img)

        start_time = time.time()
        single_cascade = self.base_cascade.do_single_cascade(
            event_img_object=target_event_obj,
            thread_id=thread_id
        )
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

    def process_frame(self, thread_id: int):
        try:
            while not self.stop_event.is_set():
                # Feed the latest image in the Queue through the cascade
                next_frame_index = self.frame_buffers.get_next_index_for_cascade()

                if next_frame_index < 0:
                    # We couldn't acquire the lock of a frame to compute the cascade; pass
                    time.sleep(0.25)
                    continue

                logger.info(f'Selected index for cascade: {next_frame_index}')
                next_frame = self.frame_buffers[next_frame_index]
                image_data = next_frame.get_img_data()
                frame_tstamp = next_frame.get_timestamp()
                self.frame_buffers.release_buffers_lock()

                total_runtime, cascade_obj = self.feed_to_cascade(
                    target_img=image_data,
                    img_name=frame_tstamp,
                    thread_id=thread_id
                )
                overhead = datetime.now(pytz.timezone('Europe/Zurich')) - frame_tstamp
                logger.debug(f'Overhead: {overhead.total_seconds()}')

                logger.debug(f"Writing cascade result of buffer # = {next_frame_index}")
                was_written = next_frame.write_cascade_data(cascade_obj, total_runtime, overhead.total_seconds())
                if was_written:
                    self.frame_buffers.mark_position_ready_for_aggregation(next_frame_index)
        except Exception as e:
            logger.exception(f"Exception in processing thread:", e)

    def single_debug(self):
        start_time = time.time()
        target_img_name = 'dummy_img.jpg'
        target_img = cv2.imread(
            os.path.join(cat_cam_py, 'readme_images/lenna_casc_Node1_001557_02_2020_05_24_09-49-35.jpg')
        )
        cascade_obj = self.feed_to_cascade(target_img=target_img, img_name=target_img_name, thread_id=-1)[1]
        current_time = time.time()
        logger.debug(f'Debug cascade runtime: {current_time - start_time}')
        return cascade_obj
