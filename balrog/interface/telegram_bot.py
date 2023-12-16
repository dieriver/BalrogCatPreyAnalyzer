import asyncio
import os
from abc import ABC, abstractmethod
from threading import Event, Thread
from typing import Callable, Self

import cv2
from cv2.typing import MatLike
from telegram import Bot, Update, ParseMode
from telegram.ext import Updater, CommandHandler
from telegram.ext.callbackcontext import CallbackContext

from balrog.config import flap_config
from balrog.utils import Logging, logger
from .flap_locker import FlapLocker


class ITelegramBot(ABC):
    def __init__(self):
        # Data coming and used form unexpected places (other files)
        self._node_live_img: MatLike | None = None
        self._node_last_casc_img: MatLike | None = None
        self._node_queue_info: int | None = None
        self._node_over_head_info: float | None = None

    @classmethod
    def get_bot_instance(
            cls,
            is_debug: bool = False,
            clean_queue_event: Event = None,
            stop_event: Event = None
    ) -> Self:
        if is_debug:
            return DebugBot()
        else:
            return BalrogTelegramBot(clean_queue_event, stop_event)

    @abstractmethod
    def send_text(self, message: str) -> None:
        pass

    @abstractmethod
    def send_img(self, img, caption) -> None:
        pass

    @property
    def node_live_img(self) -> MatLike | None:
        return self._node_live_img

    @node_live_img.setter
    def node_live_img(self, node_live_img: MatLike) -> None:
        self._node_live_img = node_live_img

    @property
    def node_last_casc_img(self) -> MatLike | None:
        return self._node_last_casc_img

    @node_last_casc_img.setter
    def node_last_casc_img(self, node_last_casc_img: MatLike) -> None:
        self._node_last_casc_img = node_last_casc_img

    @property
    def node_queue_info(self) -> int | None:
        return self._node_queue_info

    @node_queue_info.setter
    def node_queue_info(self, node_queue_info: int) -> None:
        self._node_queue_info = node_queue_info

    @property
    def node_over_head_info(self):
        return self._node_over_head_info

    @node_over_head_info.setter
    def node_over_head_info(self, node_over_head_info: float) -> None:
        self._node_over_head_info = node_over_head_info


class BalrogTelegramBot(ITelegramBot):
    def __init__(self, clean_queue_event: Event, stop_event: Event):
        # Insert Chat ID and Bot Token according to Telegram API
        super().__init__()
        if os.getenv('TELEGRAM_CHAT_ID') == "":
            raise Exception("Telegram CHAT ID not set!. Please set the 'TELEGRAM_CHAT_ID' environment variable")
        if os.getenv('TELEGRAM_BOT_TOKEN') == "":
            raise Exception("Telegram Bot token not set!. Please set the 'TELEGRAM_BOT_TOKEN' environment variable")
        self.CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
        self.BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_bot = Bot(token=self.BOT_TOKEN)
        self.bot_updater = Updater(token=self.BOT_TOKEN, use_context=True)
        self.flap_handler = FlapLocker()
        self.commands: dict[str,  Callable[[Update, CallbackContext], None]] = dict()
        pets_data = self._list_pets()
        self._populate_supported_commands(pets_data)
        # Event to signal the main loop that the queue needs to be cleaned
        self.clean_queue_event = clean_queue_event
        self.stop_event = stop_event

        # Init the listener
        self._init_bot_listener()

    def _populate_supported_commands(self, pets_data: dict[str, int]) -> None:
        self.commands['help'] = self._help_cmd_callback
        self.commands['clean'] = self._clean_cmd_callback
        self.commands['restart'] = self._restart_cmd_callback
        self.commands['nodestatus'] = self._send_status_cmd_callback
        self.commands['sendlivepic'] = self._send_live_pic_cmd_callback
        self.commands['sendlastcascpic'] = self._send_last_casc_pic_cmd_callback
        self.commands['letin'] = self._let_in_cmd_callback
        self.commands['lock'] = self._lock_moria
        self.commands['lockin'] = self._lock_moria_in
        self.commands['lockout'] = self._lock_moria_out
        self.commands['unlock'] = self._unlock_moria
        self.commands['petsstate'] = self._list_pets_state
        # create callbacks for switching the state of pets
        for name, pet_id in pets_data.items():
            self.commands[f'switch{name}'] = self._create_pet_switch_function(pet_id)
        # Not very used commands
        self.commands['curfew'] = self._set_curfew

    # Constructor supporter functions

    def _init_bot_listener(self) -> None:
        self.telegram_bot.send_message(chat_id=self.CHAT_ID, text='Balrog is online!')
        # Add all commands to handler
        for command in self.commands:
            logger.info(f"Registering command '{command}'")
            self._add_telegram_callback(command, self.commands[command])

        # Start the polling stuff
        self.bot_updater.start_polling()

    def _add_telegram_callback(self, command: str, callback: Callable[[Update, CallbackContext], None]) -> None:
        command_handler = CommandHandler(command, callback)
        self.bot_updater.dispatcher.add_handler(command_handler)

    # Raw send text and img functions

    def send_text(self, message: str) -> None:
        self.telegram_bot.send_message(chat_id=self.CHAT_ID, text=message, parse_mode=ParseMode.MARKDOWN)

    def send_img(self, img: MatLike, caption: str) -> None:
        cv2.imwrite('degubi.jpg', img)
        self.telegram_bot.send_photo(chat_id=self.CHAT_ID, photo=open('degubi.jpg', 'rb'), caption=caption)

    # Telegram Bot message handler callbacks

    def _help_cmd_callback(self, update: Update, context: CallbackContext) -> None:
        bot_message = 'Following commands supported:'
        for command in self.commands:
            bot_message += '\n /' + command
        self.send_text(bot_message)

    def _let_in_cmd_callback(self, update: Update, context: CallbackContext) -> None:
        self.send_text(f'Ok door is open for {flap_config.let_in_open_seconds}s...')
        self._unlock_moria_for_seconds(flap_config.let_in_open_seconds)
        self.clean_queue_event.set()

    def _stop_telegram(self):
        self.bot_updater.stop()
        self.bot_updater.is_idle = False

    def _restart_cmd_callback(self, update: Update, context: CallbackContext) -> None:
        self.send_text('Restarting script...')
        Thread(target=self._stop_telegram).start()
        self.stop_event.set()

    def _clean_cmd_callback(self, update: Update, context: CallbackContext) -> None:
        self.send_text('Cleaning old logs...')
        removed_paths = Logging.clean_logs()
        self.send_text(f'Removed: [{*removed_paths,}]')

    def _send_last_casc_pic_cmd_callback(self, update: Update, context: CallbackContext) -> None:
        if self.node_last_casc_img is not None:
            cv2.imwrite('last_casc.jpg', self.node_last_casc_img)
            caption = 'Last Cascade:'
            self.send_img(self.node_last_casc_img, caption)
        else:
            self.send_text('No casc img available yet...')

    def _send_live_pic_cmd_callback(self, update: Update, context: CallbackContext) -> None:
        if self.node_live_img is not None:
            cv2.imwrite('live_img.jpg', self.node_live_img)
            caption = 'Here ya go...'
            self.send_img(self.node_live_img, caption)
        else:
            self.send_text('No img available yet...')

    def _send_status_cmd_callback(self, update: Update, context: CallbackContext) -> None:
        if self.node_queue_info is not None and self.node_over_head_info is not None:
            bot_message = f'Queue length: {self.node_queue_info}\nOverhead: {self.node_over_head_info}s'
        else:
            bot_message = 'No info yet...'
        self.send_text(bot_message)

    # Internals to support the callbacks

    def _unlock_moria_for_seconds(self, seconds) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.flap_handler.unlock_for_seconds(self, seconds))
        else:
            return loop.run_until_complete(self.flap_handler.unlock_for_seconds(self, seconds))

    def _lock_moria(self, update: Update, context: CallbackContext) -> None:
        self.send_text("Locking Moria!")
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.flap_handler.lock_moria(self))
        else:
            return loop.run_until_complete(self.flap_handler.lock_moria(self))

    def _unlock_moria(self, update: Update, context: CallbackContext) -> None:
        self.send_text("Unlocking Moria!")
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.flap_handler.unlock_moria(self))
        else:
            return loop.run_until_complete(self.flap_handler.unlock_moria(self))

    def _lock_moria_in(self, update: Update, context: CallbackContext) -> None:
        self.send_text("Locking Moria for outgoing gatos!")
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.flap_handler.lock_moria_in(self))
        else:
            return loop.run_until_complete(self.flap_handler.lock_moria_in(self))

    def _lock_moria_out(self, update: Update, context: CallbackContext) -> None:
        self.send_text("Locking Moria for incoming gatos!")
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.flap_handler.lock_moria_out(self))
        else:
            return loop.run_until_complete(self.flap_handler.lock_moria_out(self))

    def _set_curfew(self, update: Update, context: CallbackContext) -> None:
        self.send_text("Activating curfew!")
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.flap_handler.activate_curfew(self))
        else:
            return loop.run_until_complete(self.flap_handler.activate_curfew(self))

    def _create_pet_switch_function(self, pet_id: int) -> Callable[[Update, CallbackContext], None]:
        def _switch_pet_state(update: Update, context: CallbackContext) -> None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(self.flap_handler.switch_pet_location(self, pet_id))
            else:
                return loop.run_until_complete(self.flap_handler.switch_pet_location(self, pet_id))
        return _switch_pet_state

    def _list_pets_state(self, update: Update, context: CallbackContext) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.flap_handler.list_pets_data(self))
        else:
            return loop.run_until_complete(self.flap_handler.list_pets_data(self))

    def _list_pets(self) -> dict[str, int]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.flap_handler.get_pets_data())
        else:
            return loop.run_until_complete(self.flap_handler.get_pets_data())


class DebugBot(ITelegramBot):
    def __init__(self):
        super().__init__()

    def send_img(self, img, caption) -> None:
        # Nothing to do here; we simply ignore the invocation
        logger.warning(f"DebugTelegramBot - Ignoring sending image!")

    def send_text(self, message: str) -> None:
        # Nothing to do here; we simply ignore the invocation
        logger.warning(f"DebugTelegramBot - Ignoring sending text!")
