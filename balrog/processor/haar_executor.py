from typing import Optional, Sequence

import cv2
from cv2.typing import MatLike

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
