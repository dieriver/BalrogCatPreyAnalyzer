from typing import Optional, Dict, Any

import numpy as np

from balrog.processor.model import configure_tensorflow
from balrog.utils.utils import logger

_model: Optional[Any] = None


def perform_pc_detection(image) -> Optional[np.ndarray]:
    if _model is None:
        return None
    return _model.predict(image)


class PCExecutor:
    def __init__(self, model_file: str, custom_objects: Dict, max_workers: int):
        configure_tensorflow(max_workers)
        self.pc_model_file_name: str = model_file
        self.custom_objects = custom_objects

    def init(self) -> None:
        global _model
        import tensorflow as tf
        _model = tf.keras.models.load_model(self.pc_model_file_name,
                                            custom_objects=self.custom_objects)
        logger.info(f"PC detection object ID: '{hex(id(_model))}'")

    def force_init(self) -> None:
        # We do nothing; this simply forces to invoke "init" to create the cascade classifier
        # for the current worker process
        logger.info(f"Starting PC detection sub-process.")