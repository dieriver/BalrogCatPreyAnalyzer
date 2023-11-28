from main_loop import FrameResultAggregator
from utils import init_logger
from config import general_config
from image_container import ImageBuffers
from camera_class import Camera


if __name__ == '__main__':
    init_logger()
    frame_buffers = ImageBuffers(2 * general_config.queue_max_threshold)

    with FrameResultAggregator(frame_buffers) as frame_aggregator:
        camera = Camera(
            fps=general_config.camera_fps,
            cleanup_threshold=general_config.camera_cleanup_frames_threshold,
            frame_buffers=frame_buffers
        )
        with camera:
            sq_cascade = FrameResultAggregator(frame_buffers)
            sq_cascade.queue_handler()
