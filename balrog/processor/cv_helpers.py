from typing import Sequence

import cv2
from cv2.typing import MatLike

from balrog.types import Box


def put_text(img: MatLike, text: str) -> MatLike:
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (153, 153, 255)
    font_scale = 1.5
    line_type = 3
    y_pos, _, _ = img.shape
    text_pos = (10, y_pos - 16)
 
    return cv2.putText(img, text, text_pos, font, font_scale, color, line_type)


def draw_rectangle(img: MatLike, box: Box, color: Sequence[float], text: str) -> MatLike:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    line_type = 3
    text_pos = (box[0][0], int(box[0][1] - 16))

    cv2.putText(img, text, text_pos, font, font_scale, color, line_type)
    return cv2.rectangle(img, (box[0][0], box[0][1]), (box[1][0], box[1][1]), color, 5)


def resize_img_to_square(img: MatLike, target_size: int, normalize: bool = False) -> MatLike:
    cv_resized = cv2.resize(img, (target_size, target_size))
    if normalize:
        # Normalize the image, so the values are floats between [0, 1]
        return cv_resized * (1. / 255)
    else:
        return cv_resized
