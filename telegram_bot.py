import asyncio
import os
from threading import Event
from typing import Callable

import cv2
from telegram import Bot, Update, ParseMode
from telegram.ext import Updater, CommandHandler
from telegram.ext.callbackcontext import CallbackContext
from telegram.ext.commandhandler import RT

from config import flap_config
from flap_locker import FlapLocker
from utils import clean_logs


class NodeBot:
    def __init__(self, clean_queue_event: Event, stop_event: Event):
        # Insert Chat ID and Bot Token according to Telegram API
        if os.getenv('TELEGRAM_CHAT_ID') == "":
            raise Exception("Telegram CHAT ID not set!. Please set the 'TELEGRAM_CHAT_ID' environment variable")
        if os.getenv('TELEGRAM_BOT_TOKEN') == "":
            raise Exception("Telegram Bot token not set!. Please set the 'TELEGRAM_BOT_TOKEN' environment variable")
        self.CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
        self.BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_bot = Bot(token=self.BOT_TOKEN)
        self.bot_updater = Updater(token=self.BOT_TOKEN, use_context=True)
        self.flap_handler = FlapLocker()
        self.commands: dict[str,  Callable[[Update, CallbackContext], RT]] = dict()
        self._populate_supported_commands()
        # Event to signal the main loop that the queue needs to be cleaned
        self.clean_queue_event = clean_queue_event
        self.stop_event = stop_event

        # Data coming and used form unexpected places (other files)
        # TODO - Directly access this data is not recommended. Refactor this!
        self.node_live_img = None
        self.node_queue_info = None
        self.node_over_head_info = None

        self.node_last_casc_img = None

        # Init the listener
        self._init_bot_listener()

    def _populate_supported_commands(self):
        self.commands['help'] = self.help_cmd_callback
        self.commands['/clean'] = self.clean_cmd_callback
        self.commands['/restart'] = self.restart_cmd_callback
        self.commands['/nodestatus'] = self.send_status_cmd_callback
        self.commands['/sendlivepic'] = self.send_status_cmd_callback
        self.commands['/sendlastcascpic'] = self.send_last_casc_pic_cmd_callback
        self.commands['/letin'] = self.let_in_cmd_callback
        self.commands['/lock'] = self._lock_moria
        self.commands['/lockin'] = self._lock_moria_in
        self.commands['/lockout'] = self._lock_moria_out
        self.commands['/curfew'] = self._set_curfew
        self.commands['/unlock'] = self._unlock_moria

    def _init_bot_listener(self):
        self.telegram_bot.send_message(chat_id=self.CHAT_ID, text='Balrog is online!')
        # Add all commands to handler
        for command in self.commands:
            self._add_telegram_callback(command, self.commands[command])

        # Start the polling stuff
        self.bot_updater.start_polling()

    def _add_telegram_callback(self, command: str, callback: Callable[[Update, CallbackContext], RT]):
        command_handler = CommandHandler(command, callback)
        self.bot_updater.dispatcher.add_handler(command_handler)

    def help_cmd_callback(self, update, context):
        bot_message = 'Following commands supported:'
        for command in self.commands:
            bot_message += '\n ' + command
        self.send_text(bot_message)

    def let_in_cmd_callback(self, update, context):
        self.send_text(f'Ok door is open for {flap_config.let_in_open_seconds}s...')
        self._unlock_moria_for_seconds(flap_config.let_in_open_seconds)
        self.clean_queue_event.set()

    def restart_cmd_callback(self, update, context):
        self.send_text('Restarting script...')
        self.stop_event.set()
        # self.send_text("(Does not work yet)")

    def clean_cmd_callback(self, update, context):
        self.send_text('Cleaning old logs...')
        removed_paths = clean_logs()
        self.send_text(f'Removed: [{*removed_paths,}]')

    def send_last_casc_pic_cmd_callback(self, update, context):
        if self.node_last_casc_img is not None:
            cv2.imwrite('last_casc.jpg', self.node_last_casc_img)
            caption = 'Last Cascade:'
            self.send_img(self.node_last_casc_img, caption)
        else:
            self.send_text('No casc img available yet...')

    def send_live_pic_cmd_callback(self, update, context):
        if self.node_live_img is not None:
            cv2.imwrite('live_img.jpg', self.node_live_img)
            caption = 'Here ya go...'
            self.send_img(self.node_live_img, caption)
        else:
            self.send_text('No img available yet...')

    def send_status_cmd_callback(self, update, context):
        if self.node_queue_info is not None and self.node_over_head_info is not None:
            bot_message = f'Queue length: {self.node_queue_info}\nOverhead: {self.node_over_head_info}s'
        else:
            bot_message = 'No info yet...'
        self.send_text(bot_message)

    # Raw send text and img functions

    def send_text(self, message: str):
        self.telegram_bot.send_message(chat_id=self.CHAT_ID, text=message, parse_mode=ParseMode.MARKDOWN)

    def send_img(self, img, caption):
        cv2.imwrite('degubi.jpg', img)
        self.telegram_bot.send_photo(chat_id=self.CHAT_ID, photo=open('degubi.jpg', 'rb'), caption=caption)

    # Internals to support the callbacks

    def _unlock_moria_for_seconds(self, seconds):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.flap_handler.unlock_for_seconds(self, seconds))
        else:
            return loop.run_until_complete(self.flap_handler.unlock_for_seconds(self, seconds))

    def _lock_moria(self, update, context):
        self.send_text("Locking Moria!")
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.flap_handler.lock_moria(self))
        else:
            return loop.run_until_complete(self.flap_handler.lock_moria(self))

    def _unlock_moria(self, update, context):
        self.send_text("Unlocking Moria!")
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.flap_handler.unlock_moria(self))
        else:
            return loop.run_until_complete(self.flap_handler.unlock_moria(self))

    def _lock_moria_in(self, update, context):
        self.send_text("Locking Moria for outgoing gatos!")
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.flap_handler.lock_moria_in(self))
        else:
            return loop.run_until_complete(self.flap_handler.lock_moria_in(self))

    def _lock_moria_out(self, update, context):
        self.send_text("Locking Moria for incoming gatos!")
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.flap_handler.lock_moria_out(self))
        else:
            return loop.run_until_complete(self.flap_handler.lock_moria_out(self))

    def _set_curfew(self, update, context):
        self.send_text("Activating curfew!")
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.flap_handler.activate_curfew(self))
        else:
            return loop.run_until_complete(self.flap_handler.activate_curfew(self))
