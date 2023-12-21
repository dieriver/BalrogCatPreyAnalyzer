from os import getenv
from threading import Event

from balrog.camera import ICamera
from balrog.config import general_config, camera_config, logging_config
from balrog.processor.image_container import ImageBuffers
from balrog.processor.main_loop import FrameResultAggregator, FrameProcessor
from balrog.utils.utils import Logging

Logging.init_logger(
    stdout_logging_level=logging_config.stdout_debug_level,
    max_log_size=logging_config.max_log_file_size_mb,
    max_log_files=logging_config.max_log_files_kept
)
stop_event = Event()
frame_buffers = ImageBuffers(2 * general_config.max_frame_buffers, logging_config.enable_circular_buffer_logging)

camera = ICamera.get_instance(
    fps=camera_config.camera_fps,
    frame_buffers=frame_buffers,
    stop_event=stop_event,
    cleanup_threshold=camera_config.camera_cleanup_frames_threshold,
    is_debug=getenv("BALROG_USE_NULL_CAMERA") is not None
)
frame_processor = FrameProcessor(frame_buffers, stop_event)
frame_aggregator = FrameResultAggregator(frame_buffers, stop_event)

with frame_aggregator, frame_processor, camera:
    frame_aggregator.aggregator_thread()
