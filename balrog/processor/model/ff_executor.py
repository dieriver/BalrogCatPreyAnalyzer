from typing import Optional

import numpy as np
import tensorflow as tf

from balrog.processor.model import configure_tensorflow
from balrog.utils.utils import logger

_model: Optional[tf.keras.Model] = None


def perform_ff_detection(image: tf.Tensor) -> Optional[np.ndarray]:
    if _model is None:
        return None
    return _model.predict(image)


class FFExecutor:
    def __init__(self, model_file: str, max_workers: int):
        configure_tensorflow(max_workers)
        self.ff_model_file_name: str = model_file

    def init(self) -> None:
        global _model
        _model = tf.keras.models.load_model(self.ff_model_file_name)
        logger.info(f"FF detection object ID: '{hex(id(_model))}'")

    def force_init(self) -> None:
        # We do nothing; this simply forces to invoke "init" to create the cascade classifier
        # for the current worker process
        logger.info(f"Starting FF detection sub-process.")