from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional

import cv2
from cv2.typing import MatLike

from balrog.interface import MessageSender
from balrog.utils import logger


def _dump_image_in_temp_file(img: MatLike) -> Optional[Path]:
    if img is None:
        return None
    temp_file = NamedTemporaryFile(delete=False, suffix=".jpg")
    cv2.imwrite(temp_file.name, img)
    return Path(temp_file.name)


def _handle_send_image(msg_sender: MessageSender, img: MatLike,
                       cumuli: float, event_str: str,
                       base_message: str, end_message: str) -> None:
    caption = f'Cumuli: {cumuli:.2f} => {base_message}{event_str}\n{end_message}'

    img_path = _dump_image_in_temp_file(img)
    if img_path is not None and caption is not None:
        msg_sender.send_img(img=Path(img_path), caption=caption)


def send_prey_message(msg_sender: MessageSender, cumuli: float, event_str: str, sender_img: MatLike) -> None:
    base_message = "PREY IN DA HOUSE!"
    end_message = ""

    logger.info(f"Sending prey message - img: {sender_img is not None}")
    _handle_send_image(msg_sender, sender_img, cumuli, event_str, base_message, end_message)


def send_no_prey_message(msg_sender: MessageSender, cumuli: float, event_str: str, sender_img: MatLike) -> None:
    base_message = "Cat is clean..."
    end_message = "Maybe use /letin?"

    logger.info(f"Sending no prey message - img: {sender_img is not None}")
    _handle_send_image(msg_sender, sender_img, cumuli, event_str, base_message, end_message)


def send_dont_know_message(msg_sender: MessageSender, cumuli: float, event_str: str, sender_img: MatLike) -> None:
    base_message = "Cant say for sure..."
    end_message = "Maybe use /letin?"

    logger.info(f"Sending don't know message - img: {sender_img is not None}")
    _handle_send_image(msg_sender, sender_img, cumuli, event_str, base_message, end_message)


def send_cat_detected_message(msg_sender: MessageSender, live_img: MatLike) -> None:
    logger.debug(f"Sending cat detected message - img: {live_img is not None}")
    img_path = _dump_image_in_temp_file(live_img)
    try:
        caption = f'Gato incoming! \nMaybe use /letin, /unlock, /lock, /lockin or /lockout?'
        msg_sender.send_img(img=img_path, caption=caption)
    except Exception:
        logger.exception('+++ Exception while sending img: ')
