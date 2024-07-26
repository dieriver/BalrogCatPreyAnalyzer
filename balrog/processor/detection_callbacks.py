from cv2.typing import MatLike

from balrog.interface import MessageSender
from balrog.utils import logger


def send_prey_message(msg_sender: MessageSender, cumuli: float, event_str: str, sender_img: MatLike) -> None:
    base_message = "PREY IN DA HOUSE!"
    end_message = ""
    caption = f'Cumuli: {cumuli} => {base_message}{event_str}\n{end_message}'

    logger.info(f"Sending prey message - img: {sender_img is not None}")
    if sender_img is not None and caption is not None:
        msg_sender.send_img(img=sender_img, caption=caption)


def send_no_prey_message(msg_sender: MessageSender, cumuli: float, event_str: str, sender_img: MatLike) -> None:
    base_message = "Cat is clean..."
    end_message = "Maybe use /letin?"
    caption = f'Cumuli: {cumuli} => {base_message}{event_str}\n{end_message}'

    logger.info(f"Sending no prey message - img: {sender_img is not None}")
    if sender_img is not None and caption is not None:
        msg_sender.send_img(img=sender_img, caption=caption)


def send_dont_know_message(msg_sender: MessageSender, cumuli: float, event_str: str, sender_img: MatLike) -> None:
    base_message = "Cant say for sure..."
    end_message = "Maybe use /letin?"
    caption = f'Cumuli: {cumuli} => {base_message}{event_str}\n{end_message}'

    logger.info(f"Sending don't know message - img: {sender_img is not None}")
    if sender_img is not None and caption is not None:
        msg_sender.send_img(img=sender_img, caption=caption)


def send_cat_detected_message(msg_sender: MessageSender, live_img: MatLike) -> None:
    logger.debug(f"Sending cat detected message - img: {live_img is not None}")
    try:
        caption = f'Gato incoming! \nMaybe use /letin, /unlock, /lock, /lockin or /lockout?'
        msg_sender.send_img(img=live_img, caption=caption)
    except Exception:
        logger.exception('+++ Exception while sending img: ')
