from cv2.typing import MatLike

from balrog.interface import ITelegramBot
from balrog.processor import EventElement
from balrog.utils import logger


def __analyze_prey_vals(
        event_objects: list[EventElement],
        cumuli: float,
        base_message: str,
        end_message: str = ''
) -> tuple[MatLike | None, str | None]:
    try:
        prey_vals = [x.pc_prey_val for x in event_objects]
        filtered_values = filter(lambda x: x is not None, prey_vals)
        if len(filtered_values) <= 0:
            logger.debug(f"Filtered all prey vals: {prey_vals}")
            return None, None
        min_prey_index = prey_vals.index(min(filtered_values))

        event_str = ''
        face_events = [x for x in event_objects if x.face_bool]
        for f_event in face_events:
            logger.debug('****************')
            logger.debug(f'Img_Name: {f_event.img_name}')
            logger.debug(f'PC_Val: {f_event.pc_prey_val:.2f}')
            logger.debug('****************')
            event_str += f'\n{f_event.img_name} => PC_Val: {f_event.pc_prey_val:.2f}'

        sender_img = event_objects[min_prey_index].output_img
        caption = f'Cumuli: {cumuli} => {base_message}{event_str}\n{end_message}'
        return sender_img, caption
    except Exception:
        logger.exception('+++ Exception while sending img: ')


def send_prey_message(bot: ITelegramBot, event_objects: list[EventElement], cumuli: float) -> None:
    logger.debug("Sending prey message")
    sender_img, caption = __analyze_prey_vals(event_objects, cumuli, 'PREY IN DA HOUSE!')
    if sender_img is not None and caption is not None:
        bot.send_img(img=sender_img, caption=caption)


def send_no_prey_message(bot: ITelegramBot, event_objects: list[EventElement], cumuli: float) -> None:
    logger.debug("Sending no prey message")
    sender_img, caption = __analyze_prey_vals(
        event_objects,
        cumuli,
        'Cat is clean...',
        'Maybe use /letin?'
    )
    if sender_img is not None and caption is not None:
        bot.send_img(img=sender_img, caption=caption)


def send_dont_know_message(bot: ITelegramBot, event_objects: list[EventElement], cumuli: float) -> None:
    logger.debug("Sending don't know message")
    sender_img, caption = __analyze_prey_vals(
        event_objects,
        cumuli,
        'Cant say for sure...',
        'Maybe use /letin?'
    )
    if sender_img is not None and caption is not None:
        bot.send_img(img=sender_img, caption=caption)


def send_cat_detected_message(bot: ITelegramBot, live_img: MatLike, cumuli: float) -> None:
    logger.debug("Sending cat detected message")
    try:
        caption = f'Cumuli: {cumuli} => Gato incoming! \nMaybe use /letin, /unlock, /lock, /lockin or /lockout?'
        # sender_img = event_objects[-1].output_img
        bot.send_img(img=live_img, caption=caption)
    except Exception:
        logger.exception('+++ Exception while sending img: ')
