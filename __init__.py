from main_loop import FrameResultAggregator, FrameProcessor
from utils import init_logger
from config import general_config
from image_container import ImageBuffers
from camera_class import Camera


if __name__ == '__main__':
    init_logger()
    frame_buffers = ImageBuffers(2 * general_config.max_frame_buffers)

    camera = Camera(
        fps=general_config.camera_fps,
        cleanup_threshold=general_config.camera_cleanup_frames_threshold,
        frame_buffers=frame_buffers
    )
    frame_processor = FrameProcessor(frame_buffers)
    frame_aggregator = FrameResultAggregator(frame_buffers)

    with camera, frame_processor, frame_aggregator:
        frame_aggregator.queue_handler()
