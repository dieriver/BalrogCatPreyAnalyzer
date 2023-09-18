import asyncio
import time
import os
import cv2
import logging
import telegram
from telegram.ext import Updater, CommandHandler
from flap_locker import FlapLocker

logger = logging.getLogger("cat_logger")


class NodeBot:
    def __init__(self):
        #Insert Chat ID and Bot Token according to Telegram API
        if os.getenv('TELEGRAM_CHAT_ID') == "":
            raise Exception("Telegram CHAT ID not set!. Please set the 'TELEGRAM_CHAT_ID' environment variable")
        if os.getenv('TELEGRAM_BOT_TOKEN') == "":
            raise Exception("Telegram Bot token not set!. Please set the 'TELEGRAM_BOT_TOKEN' environment variable")
        self.CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
        self.BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
        self.last_msg_id = 0
        self.bot_updater = Updater(token=self.BOT_TOKEN, use_context=True)
        self.bot_dispatcher = self.bot_updater.dispatcher
        self.commands = ['/help', '/nodestatus', '/sendlivepic', '/sendlastcascpic', '/letin', '/reboot', '/lock', '/lockin', '/lockout', '/curfew', '/unlock']

        self.node_live_img = None
        self.node_queue_info = None
        self.node_status = None
        self.node_last_casc_img = None
        self.node_over_head_info = None
        self.node_let_in_flag = None

        self.let_in_open_time = 40

        #Flap Locker instance
        self.flap_handler = FlapLocker()

        #Init the listener
        self.init_bot_listener()

    def init_bot_listener(self):
        telegram.Bot(token=self.BOT_TOKEN).send_message(chat_id=self.CHAT_ID, text='Good Morning, NodeBot is online!' + '🤙')
        # Add all commands to handler
        help_handler = CommandHandler('help', self.bot_help_cmd)
        self.bot_dispatcher.add_handler(help_handler)
        node_status_handler = CommandHandler('nodestatus', self.bot_send_status)
        self.bot_dispatcher.add_handler(node_status_handler)
        send_pic_handler = CommandHandler('sendlivepic', self.bot_send_live_pic)
        self.bot_dispatcher.add_handler(send_pic_handler)
        send_last_casc_pic = CommandHandler('sendlastcascpic', self.bot_send_last_casc_pic)
        self.bot_dispatcher.add_handler(send_last_casc_pic)
        letin = CommandHandler('letin', self.node_let_in)
        self.bot_dispatcher.add_handler(letin)
        reboot = CommandHandler('reboot', self.node_reboot)
        self.bot_dispatcher.add_handler(reboot)
        lock_moria = CommandHandler('lock', self.lock_moria)
        self.bot_dispatcher.add_handler(lock_moria)
        unlock_moria = CommandHandler('unlock', self.unlock_moria)
        self.bot_dispatcher.add_handler(unlock_moria)
        lock_moria_in = CommandHandler('lockin', self.lock_moria_in)
        self.bot_dispatcher.add_handler(lock_moria_in)
        lock_moria_out = CommandHandler('lockout', self.lock_moria_out)
        self.bot_dispatcher.add_handler(lock_moria_out)
        curfew = CommandHandler('curfew', self.set_curfew)
        self.bot_dispatcher.add_handler(curfew)

        # Start the polling stuff
        self.bot_updater.start_polling()

    def bot_help_cmd(self, update, context):
        bot_message = 'Following commands supported:'
        for command in self.commands:
            bot_message += '\n ' + command
        self.send_text(bot_message)

    def node_let_in(self, update, context):
        self.send_text('Ok door is open for ' + str(self.let_in_open_time) + 's...')
        self.unlock_moria_for_seconds(self.let_in_open_time)
        self.send_text('Door locked again, back to business...')
        self.node_let_in_flag = True

    def node_reboot(self, update, context):
        self.send_text('Restarting script...')
        exit(0)

    def bot_send_last_casc_pic(self, update, context):
        if self.node_last_casc_img is not None:
            cv2.imwrite('last_casc.jpg', self.node_last_casc_img)
            caption = 'Last Cascade:'
            self.send_img(self.node_last_casc_img, caption)
        else:
            self.send_text('No casc img available yet...')

    def bot_send_live_pic(self, update, context):
        if self.node_live_img is not None:
            cv2.imwrite('live_img.jpg', self.node_live_img)
            caption = 'Here ya go...'
            self.send_img(self.node_live_img, caption)
        else:
            self.send_text('No img available yet...')

    def bot_send_status(self, update, context):
        if self.node_queue_info is not None and self.node_over_head_info is not None:
            bot_message = 'Queue length: ' + str(self.node_queue_info) + '\nOverhead: ' + str(self.node_over_head_info) + 's'
        else:
            bot_message = 'No info yet...'
        self.send_text(bot_message)

    def send_text(self, message):
        telegram.Bot(token=self.BOT_TOKEN).send_message(chat_id=self.CHAT_ID, text=message, parse_mode=telegram.ParseMode.MARKDOWN)

    def send_img(self, img, caption):
        cv2.imwrite('degubi.jpg', img)
        telegram.Bot(token=self.BOT_TOKEN).send_photo(chat_id=self.CHAT_ID, photo=open('degubi.jpg', 'rb'), caption=caption)

    def unlock_moria_for_seconds(self, seconds):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.flap_handler.unlock_for_seconds(self, seconds))
        else:
            return loop.run_until_complete(self.flap_handler.unlock_for_seconds(self, seconds))

    def lock_moria(self, update, context):
        self.send_text("Locking Moria!")
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.flap_handler.lock_moria(self))
        else:
            return loop.run_until_complete(self.flap_handler.lock_moria(self))

    def unlock_moria(self, update, context):
        self.send_text("Unlocking Moria!")
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.flap_handler.unlock_moria(self))
        else:
            return loop.run_until_complete(self.flap_handler.unlock_moria(self))

    def lock_moria_in(self, update, context):
        self.send_text("Locking Moria for outgoing gatos!")
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.flap_handler.lock_moria_in(self))
        else:
            return loop.run_until_complete(self.flap_handler.lock_moria_in(self))

    def lock_moria_out(self, update, context):
        self.send_text("Locking Moria for incoming gatos!")
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.flap_handler.lock_moria_out(self))
        else:
            return loop.run_until_complete(self.flap_handler.lock_moria_out(self))

    def set_curfew(self, update, context):
        self.send_text("Activating curfew!")
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.flap_handler.activate_curfew(self))
        else:
            return loop.run_until_complete(self.flap_handler.activate_curfew(self))