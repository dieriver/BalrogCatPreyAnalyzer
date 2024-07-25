import os
from os import getenv
from signal import Signals, signal
from threading import Event

from balrog.camera import ICamera
from balrog.config import general_config, camera_config, logging_config
from balrog.interface import MessageSender
from balrog.processor.aggregator import FrameResultAggregator
from balrog.processor.frame_processor import FrameProcessor
from balrog.processor.image_container import ImageBuffers
from balrog.utils.utils import Logging


def signal_handler(sig, frame):
    message_sender.send_text("Balrog goes back to the abyss... for now...")
    message_sender.stop()


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
message_sender = MessageSender.get_message_sender_instance(
    is_debug=os.getenv("BALROG_USE_NULL_TELEGRAM") is not None,
    stop_event=stop_event
)
frame_processor = FrameProcessor(frame_buffers, stop_event)
frame_aggregator = FrameResultAggregator(frame_buffers, stop_event, message_sender)

signal(Signals.SIGTERM, signal_handler)

with frame_aggregator, frame_processor, camera:
    message_sender.start()
