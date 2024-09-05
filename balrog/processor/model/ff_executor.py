from typing import Optional, Any

import numpy as np

from balrog.utils.utils import logger

_model: Optional[Any] = None


def perform_ff_detection(image: Any) -> Optional[np.ndarray]:
    if _model is None:
        return None
    return _model.predict(image)


class FFExecutor:
    def __init__(self, model_file: str, max_workers: int):
        self.ff_model_file_name: str = model_file
        self.max_workers = max_workers

    def init(self) -> None:
        global _model
        import tensorflow as tf

        self.configure_tensorflow(tf)
        _model = tf.keras.models.load_model(self.ff_model_file_name)
        logger.info(f"FF detection object ID: '{hex(id(_model))}'")

    def force_init(self) -> None:
        # We do nothing; this simply forces to invoke "init" to create the cascade classifier
        # for the current worker process
        logger.info(f"Starting FF detection sub-process.")

    def configure_tensorflow(self, tf_module) -> None:
        # Apply TensorFlow configs, for the current process!
        tf_module.config.threading.set_inter_op_parallelism_threads(self.max_workers + 1)
        tf_module.config.threading.set_intra_op_parallelism_threads(self.max_workers * 2)

        logger.info(f"TF Config - FF: inter_threads = {tf_module.config.threading.get_inter_op_parallelism_threads()}")
        logger.info(f"TF Config - FF: intra_threads = {tf_module.config.threading.get_intra_op_parallelism_threads()}")
        logger.info(f"Num GPUs available - FF: {len(tf_module.config.list_physical_devices('GPU'))}")
