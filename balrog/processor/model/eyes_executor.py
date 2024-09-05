from typing import Optional, Any

import numpy as np

from balrog.processor.model import configure_tensorflow
from balrog.utils.utils import logger

_model: Optional[Any] = None


def perform_eye_detection(image) -> Optional[np.ndarray]:
    if _model is None:
        return None
    return _model.predict(image)


class EyesExecutor:
    def __init__(self, model_file: str, max_workers: int):
        configure_tensorflow(max_workers)
        self.eyes_model_file_name: str = model_file

    def init(self) -> None:
        global _model
        import tensorflow as tf
        _model = tf.keras.models.load_model(self.eyes_model_file_name)
        logger.info(f"Eyes detection object ID: '{hex(id(_model))}'")

    def force_init(self) -> None:
        # We do nothing; this simply forces to invoke "init" to create the cascade classifier
        # for the current worker process
        logger.info(f"Starting Eyes detection sub-process.")