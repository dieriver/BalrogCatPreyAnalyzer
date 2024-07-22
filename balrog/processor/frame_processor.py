import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from multiprocessing import Event
from typing import Optional

import cv2
import pytz
from cv2.typing import MatLike

from balrog.config import general_config, logging_config, camera_config
from balrog.processor import Cascade, EventElement
from balrog.processor.image_container import ImageBuffers, ImageContainer
from balrog.utils import logger, get_resource_path


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
        self.frame_processor_pool = ThreadPoolExecutor(max_workers=general_config.max_frame_processor_threads)

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
                next_frame_index, next_frame_copy = self.frame_buffers.get_next_index_for_cascade()

                if next_frame_index < 0 or next_frame_copy is None:
                    # We couldn't acquire the lock of a frame to compute the cascade; pass
                    logger.debug(f"Could not get nex_frame_index: {next_frame_index}")
                    time.sleep(3 * 1 / camera_config.camera_fps)
                    continue

                logger.debug(f'Thread {thread_id} - Index for cascade: {next_frame_index}')

                total_runtime, cascade_obj = self.feed_to_cascade(
                    target_img=next_frame_copy.img_data,
                    img_name=next_frame_copy.timestamp.strftime(general_config.timestamp_format),
                    thread_id=thread_id,
                    frame_index=next_frame_index
                )
                overhead = datetime.now(pytz.timezone(general_config.local_timezone)) - next_frame_copy.timestamp
                logger.debug(f'Thread {thread_id} - Overhead: {overhead.total_seconds()}')

                logger.debug(f"Thread {thread_id} - Writing cascade result of buffer # = {next_frame_index}")
                self.frame_buffers.write_cascade_data(
                    next_frame_index,
                    cascade_obj,
                    total_runtime,
                    overhead.total_seconds()
                )
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
            target_img = cv2.imread(str(resource.resolve()))
        cascade_obj = self.feed_to_cascade(target_img=target_img, img_name=target_img_name)[1]
        current_time = time.time()
        logger.debug(f'Debug cascade runtime: {current_time - start_time}')
        return cascade_obj
