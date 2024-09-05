from typing import Optional, Any, Dict

from balrog.processor.model import configure_tensorflow
from balrog.utils.utils import logger

_model: Optional[Any] = None


def perform_cc_mobile_detection(image) -> Dict[str, Any]:
    if _model is None:
        return {}
    return _model(image)


class CCMobileExecutor:
    def __init__(self, cc_mobile_model_file_name: str, max_workers: int):
        configure_tensorflow(max_workers)
        self.cc_mobile_model_file_name: str = cc_mobile_model_file_name

    def init(self) -> None:
        global _model
        import tensorflow as tf
        _detect_function = tf.saved_model.load(self.cc_mobile_model_file_name)
        logger.info(f"CC Mobile detection object ID: '{hex(id(_detect_function))}'")

    def force_init(self) -> None:
        # We do nothing; this simply forces to invoke "init" to create the cascade classifier
        # for the current worker process
        logger.info(f"Starting CC Mobile detection sub-process.")