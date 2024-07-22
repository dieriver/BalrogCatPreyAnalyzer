import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Event
from typing import Tuple, Optional, List

from cv2.typing import MatLike

from balrog.config import general_config, model_config, camera_config
from balrog.interface import MessageSender
from balrog.processor import ImageBuffers, EventElement
from balrog.processor.detection_callbacks import send_cat_detected_message, send_no_prey_message, send_prey_message, \
    send_dont_know_message
from balrog.utils import logger


def _get_min_prey_tuple(events: List[EventElement]) -> Tuple[int, float]:
    minimum: float = sys.float_info.max
    min_index: int = -1
    for index, event in enumerate(events):
        if event.pc_prey_val is not None and event.pc_prey_val < minimum:
            minimum = event.pc_prey_val
            min_index = index
    return min_index, minimum


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
        self.stop_event = stop_event
        self.bot = MessageSender.get_message_sender_instance(
            is_debug=os.getenv("BALROG_USE_NULL_TELEGRAM") is not None,
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
            logger.error(f"Traceback: {traceback.format_tb(tb)}")
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
                    logger.debug(f"Not enough frames ready for aggregation: {frames_rdy_for_aggregation}")
                    time.sleep(3 * 1 / camera_config.camera_fps)
            except Exception as e:
                logger.exception("Exception in aggregation thread: ", e)
                logger.info("Cleaning queue since exception")
                self.frame_buffers.clear()

    def aggregate_available_frames(self, frames_rdy_for_aggregation: int):
        # We get the last buffer, and extract its data
        next_frame_index, next_frame = self.frame_buffers.get_next_index_for_aggregation()
        if next_frame_index < 0 or next_frame is None:
            return

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
            if 0 < model_config.cat_counter_threshold <= self.cat_counter and not self.CAT_DETECTED_FLAG:
                self.CAT_DETECTED_FLAG = True
                send_cat_detected_message(self.bot, image_data)

            # Last cat pic for bot
            self.bot.node_last_casc_img = cascade_obj.output_img

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
                    cumuli_cpy = self.cumulus_points / self.face_counter
                    image, event_str = self._analyze_prey_vals()
                    self.verdict_sender_pool.submit(
                        send_no_prey_message,
                        self.bot, cumuli_cpy, event_str, image
                    )
                    self.reset_aggregation_fields()

                elif self.cumulus_points / self.face_counter < model_config.cumulus_prey_threshold:
                    self.PREY_FLAG = True
                    logger.info('**** IT IS A PREY!!!!! ****')
                    cumuli_cpy = self.cumulus_points / self.face_counter
                    image, event_str = self._analyze_prey_vals()
                    self.verdict_sender_pool.submit(
                        send_prey_message,
                        self.bot, cumuli_cpy, event_str, image
                    )
                    self.reset_aggregation_fields()
                else:
                    self.NO_PREY_FLAG = False
                    self.PREY_FLAG = False

            # Cat was found => still belongs to event => acts as dk state
            self.event_reset_counter = 0
        else:
            # No cat detected => reset event_counters if necessary
            logger.info('**** NO CAT FOUND! ****')
            self.event_reset_counter += 1
            if self.event_reset_counter >= model_config.event_reset_threshold:
                # If was True => event now over => clear queue
                if self.EVENT_FLAG:
                    cumuli_cpy = self.cumulus_points / (1 if self.face_counter == 0 else self.face_counter)
                    image, event_str = self._analyze_prey_vals()
                    self.verdict_sender_pool.submit(
                        send_dont_know_message,
                        self.bot, cumuli_cpy, event_str, image
                    )
                self.reset_aggregation_fields()
                logger.debug(f'--- EVENT ENDED: {self.event_reset_counter} > {model_config.event_reset_threshold} ---')

        if self.EVENT_FLAG and self.FACE_FOUND_FLAG:
            self.patience_counter += 1
        if self.patience_counter > 2 or self.face_counter > 1:
            self.PATIENCE_FLAG = True

    def _analyze_prey_vals(
            self
    ) -> Tuple[Optional[MatLike], Optional[str]]:
        min_prey_index = None
        try:
            min_prey_index, _ = _get_min_prey_tuple(self.event_objects)

            if min_prey_index < 0:
                logger.warning(f"No minimal index & value found in: {[x.pc_prey_val for x in self.event_objects]}")
                return None, None

            event_str = ''
            face_events = [x for x in self.event_objects if x.face_bool]
            for f_event in face_events:
                logger.debug('****************')
                logger.debug(f'Img_Name: {f_event.img_name}')
                logger.debug(f'PC_Val: {f_event.pc_prey_val:.2f}')
                logger.debug('****************')
                event_str += f'\n{f_event.img_name} => PC_Val: {f_event.pc_prey_val:.2f}'

            sender_img = self.event_objects[min_prey_index].output_img
            return sender_img, event_str
        except Exception:
            logger.info(f"min_prey_index = {min_prey_index}, event_size = {len(self.event_objects)}")
            logger.exception('+++ Exception while sending img: ')
            return None, None
