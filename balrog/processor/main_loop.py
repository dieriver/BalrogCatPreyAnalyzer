import copy
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from multiprocessing import Event
from typing import Optional

import cv2
import pytz
from cv2.typing import MatLike

from balrog.config import general_config, model_config, logging_config
from balrog.interface import MessageSender
from balrog.processor import Cascade, EventElement
from balrog.processor.image_container import ImageBuffers, ImageContainer
from balrog.utils import logger, get_resource_path
from .detection_callbacks import (
    send_cat_detected_message,
    send_dont_know_message,
    send_prey_message,
    send_no_prey_message
)


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
        self.bot = MessageSender.get_message_sender_instance(
            is_debug=os.getenv("BALROG_USE_NULL_TELEGRAM") is not None,
            clean_queue_event=self.clean_queue_event,
            stop_event=stop_event
        )
        self.verdict_sender_pool = ThreadPoolExecutor(max_workers=general_config.max_message_sender_threads)
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
        self.event_objects: list[EventElement] = []
        self.frame_buffers = frame_buffers

    def __enter__(self):
        # We don't do anything here
        pass

    def __exit__(self, exception_type, exception_value, tb):
        self.verdict_sender_pool.shutdown(wait=False, cancel_futures=True)
        if exception_type is not None:
            logger.error(f"Something wrong happened in the frame result aggregator thread")
            logger.error(f"Exception type: {repr(exception_type)}")
        if exception_value is not None:
            logger.error(f"Exception value: {exception_value}")
        if tb is not None:
            logger.error(f"Traceback: {''.join(traceback.format_tb(tb))}")
            sys.exit(1)
        # We use a "successful" exit code to restart the script
        # This is interpreted as a call to restart the script
        sys.exit(0)

    def reset_aggregation_fields(self):
        # TODO - Do not rely on this "static" state that needs to be reset every time we reach a verdict
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
        self.clean_queue_event.clear()
        # The next operation is expensive, maybe we don't need to perform it every single time
        #self.frame_buffers.clear()

    def aggregator_thread(self):
        while not self.stop_event.is_set():
            try:
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
                logger.info("Cleaning queue since exception")
                self.frame_buffers.clear()

    def aggregate_available_frames(self, frames_rdy_for_aggregation: int):
        # We get the last buffer, and extract its data
        next_frame_index = self.frame_buffers.get_next_index_for_aggregation()
        if next_frame_index < 0:
            return

        next_frame = self.frame_buffers[next_frame_index].clone()
        # We release the lock asap
        self.frame_buffers.reset_buffer(next_frame_index)

        cascade_obj: EventElement = next_frame.event_element
        overhead: float = next_frame.overhead
        image_data: MatLike = next_frame.img_data

        # Add this such that the bot has some info
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
                self.verdict_sender_pool.submit(send_cat_detected_message, self.bot, node_live_img_cpy, 0)

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
                    #events_cpy = copy.deepcopy(self.event_objects)
                    cumuli_cpy = self.cumulus_points / self.face_counter
                    self.verdict_sender_pool.submit(
                        send_no_prey_message,
                        self.bot, copy.deepcopy(self.event_objects), cumuli_cpy
                    )
                    self.reset_aggregation_fields()
                elif self.cumulus_points / self.face_counter < model_config.cumulus_prey_threshold:
                    self.PREY_FLAG = True
                    logger.info('**** IT IS A PREY!!!!! ****')
                    events_cpy = copy.deepcopy(self.event_objects)
                    cumuli_cpy = self.cumulus_points / self.face_counter
                    self.verdict_sender_pool.submit(
                        send_prey_message,
                        self.bot, events_cpy, cumuli_cpy
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
                    # TODO QUICK FIX
                    if self.face_counter == 0:
                        self.face_counter = 1
                    #events_cpy = copy.deepcopy(self.event_objects)
                    cumuli_cpy = self.cumulus_points / self.face_counter
                    self.verdict_sender_pool.submit(
                        send_dont_know_message,
                        self.bot, copy.deepcopy(self.event_objects), cumuli_cpy
                    )
                logger.debug(f'---- CLEARED QUEUE BECAUSE EVENT ENDED: {self.event_reset_counter} > {model_config.event_reset_threshold} ----')
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
        self.frame_processor_pool = ThreadPoolExecutor(max_workers=general_config.max_message_sender_threads)

    def __enter__(self):
        # Do this to force run all networks s.t. the network inference time stabilizes
        self.single_debug()
        # We need to submit the process tasks here
        for i in range(0, general_config.max_frame_processor_threads):
            self.frame_processor_pool.submit(self.process_frame, i)

    def __exit__(self, exception_type, exception_value, tb):
        self.frame_processor_pool.shutdown(wait=False, cancel_futures=True)
        if exception_type is not None:
            logger.error(f"Something wrong happened in the frame processor thread")
            logger.error(f"Exception type: {exception_type}")
        if exception_value is not None:
            logger.error(f"Exception value: {exception_value}")
        if tb is not None:
            logger.error(f"Traceback: {''.join(traceback.format_tb(tb))}")
        return True

    def feed_to_cascade(self, target_img: MatLike, img_name: str, thread_id: int = -1, frame_index: int = -1) -> tuple[float, EventElement]:
        target_event_obj = EventElement(img_name=img_name, cc_target_img=target_img)

        start_time = time.time()
        self.base_cascade.do_single_cascade(
            event_img_object=target_event_obj,
            thread_id=thread_id,
            frame_index=frame_index
        )
        target_event_obj.total_inference_time = sum(filter(None, [
            target_event_obj.cc_inference_time,
            target_event_obj.cr_inference_time,
            target_event_obj.bbs_inference_time,
            target_event_obj.haar_inference_time,
            target_event_obj.ff_bbs_inference_time,
            target_event_obj.ff_haar_inference_time,
            target_event_obj.pc_inference_time]))
        total_runtime = time.time() - start_time
        logger.debug(f'Thread {thread_id} - Total Runtime: {total_runtime}')

        return total_runtime, target_event_obj

    def process_frame(self, thread_id: int) -> None:
        next_frame_copy: Optional[ImageContainer] = None
        while not self.stop_event.is_set():
            try:
                # Feed the latest image in the Queue through the cascade
                next_frame_index = self.frame_buffers.get_next_index_for_cascade()

                if next_frame_index < 0:
                    # We couldn't acquire the lock of a frame to compute the cascade; pass
                    time.sleep(0.25)
                    continue

                logger.debug(f'Thread {thread_id} - Index for cascade: {next_frame_index}')
                next_frame_copy = self.frame_buffers[next_frame_index].clone()

                total_runtime, cascade_obj = self.feed_to_cascade(
                    target_img=next_frame_copy.img_data,
                    img_name=next_frame_copy.timestamp.strftime(general_config.timestamp_format),
                    thread_id=thread_id,
                    frame_index=next_frame_index
                )
                overhead = datetime.now(pytz.timezone(general_config.local_timezone)) - next_frame_copy.timestamp
                logger.debug(f'Thread {thread_id} - Overhead: {overhead.total_seconds()}')

                logger.debug(f"Thread {thread_id} - Writing cascade result of buffer # = {next_frame_index}")
                self.frame_buffers.write_cascade_data(next_frame_index, cascade_obj, total_runtime, overhead.total_seconds())
            except Exception:
                if next_frame_copy is not None:
                    img_name = next_frame_copy.timestamp.strftime(general_config.timestamp_format)
                    filename = f'{logging_config.log_dbg_img_folder}/{img_name.replace(" ", "_")}.jpg'
                    cv2.imwrite(
                        filename,
                        next_frame_copy.img_data
                    )
                logger.exception(f"Thread {thread_id} - Exception in processing thread:")
                logger.info(f"Thread {thread_id} - Cleaning queue since exception")
                self.frame_buffers.clear()


    def single_debug(self):
        start_time = time.time()
        target_img_name = 'dummy_img.jpg'
        with get_resource_path("dbg_casc.jpg") as resource:
            target_img = cv2.imread(
                str(resource.resolve())
            )
        cascade_obj = self.feed_to_cascade(target_img=target_img, img_name=target_img_name)[1]
        current_time = time.time()
        logger.debug(f'Debug cascade runtime: {current_time - start_time}')
        return cascade_obj
