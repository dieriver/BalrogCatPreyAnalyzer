from typing import Optional, Any, Dict

from balrog.utils.utils import logger

_model: Optional[Any] = None


def perform_cc_mobile_detection(image) -> Dict[str, Any]:
    if _model is None:
        return {}
    return _model(image)


class CCMobileExecutor:
    def __init__(self, cc_mobile_model_file_name: str, max_workers: int):
        self.cc_mobile_model_file_name: str = cc_mobile_model_file_name
        self.max_workers = max_workers

    def init(self) -> None:
        global _model
        import tensorflow as tf

        self.configure_tensorflow(tf)
        _detect_function = tf.saved_model.load(self.cc_mobile_model_file_name)
        logger.info(f"CC Mobile detection object ID: '{hex(id(_detect_function))}'")

    def force_init(self) -> None:
        # We do nothing; this simply forces to invoke "init" to create the cascade classifier
        # for the current worker process
        logger.info(f"Starting CC Mobile detection sub-process.")

    def configure_tensorflow(self, tf_module) -> None:
        # Apply TensorFlow configs, for the current process!
        tf_module.config.threading.set_inter_op_parallelism_threads(self.max_workers + 1)
        tf_module.config.threading.set_intra_op_parallelism_threads(self.max_workers * 2)

        logger.info(f"TF Config - PC: inter_threads = {tf_module.config.threading.get_inter_op_parallelism_threads()}")
        logger.info(f"TF Config - PC: intra_threads = {tf_module.config.threading.get_intra_op_parallelism_threads()}")
        logger.info(f"Num GPUs available: {len(tf_module.config.list_physical_devices('GPU'))}")
