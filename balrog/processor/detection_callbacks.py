from balrog.interface import ITelegramBot
from balrog.utils import logger


def __analyze_prey_vals(event_objects, cumuli, base_message, end_message=''):
    try:
        prey_vals = [x.pc_prey_val for x in event_objects]
        min_prey_index = prey_vals.index(min(filter(lambda x: x is not None, prey_vals)))

        event_str = ''
        face_events = [x for x in event_objects if x.face_bool]
        for f_event in face_events:
            logger.debug('****************')
            logger.debug(f'Img_Name: {f_event.img_name}')
            logger.debug(f'PC_Val: {f_event.pc_prey_val:.2f}')
            logger.debug('****************')
            event_str += f'\n{f_event.img_name} => PC_Val: {f_event.pc_prey_val:.2f}'

        sender_img = event_objects[min_prey_index].output_img
        caption = f'Cumuli: {cumuli} => {base_message}{event_str}{end_message}'
        return sender_img, caption
    except Exception:
        logger.exception('+++ Exception while sending img: ')


def send_prey_message(bot: ITelegramBot, event_objects, cumuli):
    sender_img, caption = __analyze_prey_vals(event_objects, cumuli, 'PREY IN DA HOUSE!')
    bot.send_img(img=sender_img, caption=caption)


def send_no_prey_message(bot: ITelegramBot, event_objects, cumuli):
    sender_img, caption = __analyze_prey_vals(
        event_objects,
        cumuli,
        'Cat is clean...',
        '\nMaybe use /letin?'
    )
    bot.send_img(img=sender_img, caption=caption)


def send_dk_message(bot: ITelegramBot, event_objects, cumuli):
    sender_img, caption = __analyze_prey_vals(
        event_objects,
        cumuli,
        'Cant say for sure...',
        '\nMaybe use /letin?'
    )
    bot.send_img(img=sender_img, caption=caption)


def send_cat_detected_message(bot: ITelegramBot, live_img, cumuli):
    try:
        caption = f'Cumuli: {cumuli} => Gato incoming! \nMaybe use /letin, /unlock, /lock, /lockin or /lockout?'
        # sender_img = event_objects[-1].output_img
        bot.send_img(img=live_img, caption=caption)
    except Exception:
        logger.exception('+++ Exception while sending img: ')
