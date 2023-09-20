from telegram_bot import NodeBot
from utils import logger


def send_prey_message(bot: NodeBot, event_objects, cumuli):
    try:
        prey_vals = [x.pc_prey_val for x in event_objects]
        max_prey_index = prey_vals.index(max(filter(lambda x: x is not None, prey_vals)))

        event_str = ''
        face_events = [x for x in event_objects if x.face_bool]
        for f_event in face_events:
            logger.debug('****************')
            logger.debug('Img_Name:' + str(f_event.img_name))
            logger.debug('PC_Val:' + str('%.2f' % f_event.pc_prey_val))
            logger.debug('****************')
            event_str += '\n' + f_event.img_name + ' => PC_Val: ' + str('%.2f' % f_event.pc_prey_val)

        sender_img = event_objects[max_prey_index].output_img
        caption = 'Cumuli: ' + str(cumuli) + ' => PREY IN DA HOUSE!' + ' 🐁🐁🐁' + event_str
        bot.send_img(img=sender_img, caption=caption)
    except Exception:
        logger.exception('+++ Exception while sending img: ')


def send_no_prey_message(bot: NodeBot, event_objects, cumuli):
    try:
        prey_vals = [x.pc_prey_val for x in event_objects]
        min_prey_index = prey_vals.index(min(filter(lambda x: x is not None, prey_vals)))

        event_str = ''
        face_events = [x for x in event_objects if x.face_bool]
        for f_event in face_events:
            logger.debug('****************')
            logger.debug('Img_Name:' + str(f_event.img_name))
            logger.debug('PC_Val:' + str('%.2f' % f_event.pc_prey_val))
            logger.debug('****************')
            event_str += '\n' + f_event.img_name + ' => PC_Val: ' + str('%.2f' % f_event.pc_prey_val)

        sender_img = event_objects[min_prey_index].output_img
        caption = 'Cumuli: ' + str(cumuli) + ' => Cat is clean...' + ' 🐱' + event_str
        bot.send_img(img=sender_img, caption=caption)
    except Exception:
        logger.exception('+++ Exception while sending img: ')


def send_dk_message(bot: NodeBot, event_objects, cumuli):
    try:
        event_str = ''
        face_events = [x for x in event_objects if x.face_bool]
        for f_event in face_events:
            logger.debug('****************')
            logger.debug('Img_Name:' + str(f_event.img_name))
            logger.debug('PC_Val:' + str('%.2f' % f_event.pc_prey_val))
            logger.debug('****************')
            event_str += '\n' + f_event.img_name + ' => PC_Val: ' + str('%.2f' % f_event.pc_prey_val)

        sender_img = face_events[0].output_img
        caption = 'Cumuli: ' + str(cumuli) + ' => Cant say for sure...' + ' 🤷‍♀️' + event_str + '\nMaybe use /letin?'
        bot.send_img(img=sender_img, caption=caption)
    except Exception:
        logger.exception('+++ Exception while sending img: ')


def send_cat_detected_message(bot: NodeBot, live_img, cumuli):
    try:
        caption = 'Cumuli: ' + str(
            cumuli) + ' => Gato incoming! 🐱🐈🐱' + '\nMaybe use /letin, /unlock, /lock, /lockin or /lockout?'
        # sender_img = event_objects[-1].output_img
        bot.send_img(img=live_img, caption=caption);
    except Exception:
        logger.exception('+++ Exception while sending img: ')