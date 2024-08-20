from typing import Optional, Sequence

import cv2
from cv2.typing import MatLike

from balrog.utils.utils import logger

# This is the CascadeClassifier instance (OpenCV object) _of the current worker_.
# Since the main process will be forked, each sub-process should have a _different_
# instance of the CascadeClassifier, tied to a _different instance of the dynamic
# link of the OpenCV python bindings_. This avoids static state corruption when
# invoking the detection from different threads of the same process.
_model: Optional[cv2.CascadeClassifier] = None


def perform_haar_detection(image: MatLike) -> Sequence[cv2.typing.Rect]:
    global _model
    if _model is None:
        return []
    return _model.detectMultiScale(image=image, scaleFactor=1.3, minNeighbors=1, minSize=(25, 25))


class HaarExecutor:
    def __init__(self, haar_model_file_name: str):
        self.haar_model_file_name: str = haar_model_file_name

    def init(self) -> None:
        global _model
        _model = cv2.CascadeClassifier(self.haar_model_file_name)

    def force_init(self) -> None:
        # We do nothing; this simply forces to invoke "init" to create the cascade classifier
        # for the current worker process
        logger.info(f"Starting HAAR detection sub-process. Haar object id: {id(_model)}")
