from typing import Optional, Dict, Any

import numpy as np

from balrog.utils.utils import logger

_model: Optional[Any] = None


def perform_pc_detection(image) -> Optional[np.ndarray]:
    if _model is None:
        return None
    return _model.predict(image)


class PCExecutor:
    def __init__(self, model_file: str, custom_objects: Dict, max_workers: int):
        self.pc_model_file_name: str = model_file
        self.custom_objects = custom_objects
        self.max_workers = max_workers

    def init(self) -> None:
        global _model
        import tensorflow as tf

        self.configure_tensorflow(tf)
        _model = tf.keras.models.load_model(self.pc_model_file_name,
                                            custom_objects=self.custom_objects)
        logger.info(f"PC detection object ID: '{hex(id(_model))}'")

    def force_init(self) -> None:
        # We do nothing; this simply forces to invoke "init" to create the cascade classifier
        # for the current worker process
        logger.info(f"Starting PC detection sub-process.")

    def configure_tensorflow(self, tf_module) -> None:
        # Apply TensorFlow configs, for the current process!
        tf_module.config.threading.set_inter_op_parallelism_threads(self.max_workers + 1)
        tf_module.config.threading.set_intra_op_parallelism_threads(self.max_workers * 2)

        logger.info(f"TF Config - PC: inter_threads = {tf_module.config.threading.get_inter_op_parallelism_threads()}")
        logger.info(f"TF Config - PC: intra_threads = {tf_module.config.threading.get_intra_op_parallelism_threads()}")
        logger.info(f"Num GPUs available: {len(tf_module.config.list_physical_devices('GPU'))}")
