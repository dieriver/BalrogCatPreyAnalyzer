import tensorflow as tf

from balrog.utils import logger


def configure_tensorflow(tf_max_threads) -> None:
    # Apply TensorFlow configs, for the current process!
    tf.config.threading.set_inter_op_parallelism_threads(tf_max_threads + 1)
    tf.config.threading.set_intra_op_parallelism_threads(tf_max_threads * 2)

    logger.info(f"TF Config: inter_threads = {tf.config.threading.get_inter_op_parallelism_threads()}")
    logger.info(f"TF Config: intra_threads = {tf.config.threading.get_intra_op_parallelism_threads()}")
    logger.info(f"Num GPUs available: {len(tf.config.list_physical_devices('GPU'))}")


from .cc_mobile_executor import CCMobileExecutor, perform_cc_mobile_detection
from .haar_executor import HaarModelExecutor, perform_haar_detection
from .pc_executor import PCExecutor, perform_pc_detection
from .ff_executor import FFExecutor, perform_ff_detection
from .eyes_executor import EyesExecutor, perform_eye_detection
