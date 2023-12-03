from threading import Event
from main_loop import FrameResultAggregator, FrameProcessor
from utils import Logging
from config import general_config, camera_config, logging_config
from image_container import ImageBuffers
from camera_class import Camera


if __name__ == '__main__':
    Logging.init_logger(
        stdout_logging_level=logging_config.stdout_debug_level,
        max_log_size=logging_config.max_log_file_size_mb,
        max_log_files=logging_config.max_log_files_kept
    )
    stop_event = Event()
    frame_buffers = ImageBuffers(2 * general_config.max_frame_buffers, logging_config.enable_circular_buffer_logging)

    camera = Camera(
        fps=camera_config.camera_fps,
        cleanup_threshold=camera_config.camera_cleanup_frames_threshold,
        frame_buffers=frame_buffers,
        stop_event=stop_event
    )
    frame_processor = FrameProcessor(frame_buffers, stop_event)
    frame_aggregator = FrameResultAggregator(frame_buffers, stop_event)

    with camera, frame_processor, frame_aggregator:
        frame_aggregator.queue_handler()
