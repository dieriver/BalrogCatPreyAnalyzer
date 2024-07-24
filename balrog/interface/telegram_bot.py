import asyncio
import os
from tempfile import TemporaryDirectory
from threading import Event
from typing import Any, Callable, Dict, TypeVar, Coroutine

import cv2
from cv2.typing import MatLike
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

from balrog.config import flap_config, general_config, command_aliases_config
from balrog.interface import MessageSender
from balrog.interface.flap_locker import FlapLocker
from balrog.utils import Logging, logger

_T = TypeVar("_T")
TelegramCallbackType = Callable[[Update, ContextTypes.DEFAULT_TYPE], Coroutine[Any, Any, None]]


class BalrogTelegramBot(MessageSender):
    def __init__(self, stop_event: Event):
        # Insert Chat ID and Bot Token according to Telegram API
        super().__init__()
        if os.getenv('TELEGRAM_CHAT_ID') == "":
            raise Exception("Telegram CHAT ID not set!. Please set the 'TELEGRAM_CHAT_ID' environment variable")
        if os.getenv('TELEGRAM_BOT_TOKEN') == "":
            raise Exception("Telegram Bot token not set!. Please set the 'TELEGRAM_BOT_TOKEN' environment variable")
        self.stop_event = stop_event
        self.CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
        self.BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_endpoint = Application.builder().token(self.BOT_TOKEN).build()
        self.flap_handler = FlapLocker()
        self.commands: Dict[str, TelegramCallbackType] = dict()
        pets_data = asyncio.run(self.flap_handler.get_pets_data())
        devices_data = asyncio.run(self.flap_handler.get_devices_data())
        # Since asyncio closes the event loop, we need to re-open it for the polling

        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)

        self._populate_supported_commands(pets_data, devices_data)
        self._populate_command_aliases()
        self.is_ongoing_let_in: bool = False

        # Init the listener
        self._init_bot_listener()

    def _populate_command_aliases(self):
        for command in command_aliases_config.aliases_map:
            for alias in command_aliases_config.aliases_map[command]:
                self.commands[alias] = self.commands[command]

    def _populate_supported_commands(self, pets_data: Dict[str, int], devices_data: Dict[str, int]) -> None:
        self.commands['help'] = self._get_help_cmd_callback()
        self.commands['clean'] = self._get_clean_cmd_callback()
        self.commands['restart'] = self._get_restart_cmd_callback()
        self.commands['nodestatus'] = self._get_node_status_cmd_callback()
        self.commands['sendlivepic'] = self._get_send_live_pic_cmd_callback()
        self.commands['sendlastcascpic'] = self._get_send_last_casc_pic_cmd_callback()
        self.commands['letin'] = self._get_let_in_callback()
        self.commands['cancelLetin'] = self._get_cancel_let_in_callback()
        self.commands['lock'] = self._get_lock_moria_callback()
        self.commands['lockin'] = self._get_lock_moria_in_callback()
        self.commands['lockout'] = self._get_lock_moria_out_callback()
        self.commands['unlock'] = self._get_unlock_moria_callback()
        self.commands['statusPets'] = self._get_status_pets_callback()
        self.commands['mute'] = self._get_mute_notifications_callback()
        # create callbacks for switching the state of pets
        for name, pet_id in pets_data.items():
            self.commands[f'switch{name}'] = self._get_switch_pet_location_callback(pet_id)
        # create callbacks for status of the devices
        for name, device_id in devices_data.items():
            self.commands[f'status{name}'] = self._get_send_device_data_callback(device_id)
        # Not very used commands
        self.commands['curfew'] = self._get_activate_curfew_callback()

    # Constructor supporter functions

    def _init_bot_listener(self) -> None:
        self.send_text('Balrog is online!')
        # Add all commands to handler
        handlers = []
        for command in self.commands:
            logger.info(f"Registering command '{command}'")
            handlers.append(CommandHandler(command, self.commands[command]))
        self.telegram_endpoint.add_handlers(handlers)

        # Start the polling stuff
        self.telegram_endpoint.run_polling(allowed_updates=[Update.MESSAGE])

    # Raw send text and img functions

    def send_text(self, message: str) -> None:
        self.telegram_endpoint.bot.send_message(
            chat_id=self.CHAT_ID,
            text=message,
            parse_mode=ParseMode.MARKDOWN_V2
        )

    def send_img(self, img: MatLike, caption: str, force_send: bool = False) -> None:
        if not force_send and self.muted_images:
            return
        with TemporaryDirectory() as tmp_dir:
            cv2.imwrite(f'{tmp_dir}/balrog_send_img.jpg', img)
            self.telegram_endpoint.bot.send_photo(
                chat_id=self.CHAT_ID,
                photo=open(f'{tmp_dir}/balrog_send_img.jpg', 'rb'),
                caption=caption
            )

    def _get_help_cmd_callback(self) -> TelegramCallbackType:
        bot = self

        async def help_cmd_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            nonlocal bot
            bot_message = 'Following commands supported:'
            for command in bot.commands:
                bot_message += '\n /' + command
            await update.message.reply_text(bot_message)
        return help_cmd_callback

    def _get_clean_cmd_callback(self) -> TelegramCallbackType:
        async def clean_cmd_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            await update.message.reply_text('Cleaning old logs...')
            removed_paths = Logging.clean_logs()
            await update.message.reply_text(f'Removed: [{*removed_paths,}]')
        return clean_cmd_callback

    def _get_restart_cmd_callback(self) -> TelegramCallbackType:
        bot = self

        async def _restart_cmd_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            nonlocal bot
            await update.message.reply_text('Restarting script...')
            bot.telegram_endpoint.stop()
            bot.telegram_endpoint.is_idle = False
            bot.stop_event.set()
        return _restart_cmd_callback

    def _get_node_status_cmd_callback(self) -> TelegramCallbackType:
        bot = self

        async def _node_status_cmd_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            nonlocal bot
            if bot.node_queue_info is not None and bot.node_over_head_info is not None:
                bot_message = f'Queue length: {bot.node_queue_info}\nOverhead: {bot.node_over_head_info}s'
            else:
                bot_message = 'No info yet...'
            await update.message.reply_text(bot_message)
        return _node_status_cmd_callback

    def _get_send_live_pic_cmd_callback(self) -> TelegramCallbackType:
        bot = self

        async def _send_live_pic_cmd_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            nonlocal bot
            if bot.node_live_img is not None:
                caption = 'Here it is...'
                await update.message.reply_photo(bot.node_live_img, caption)
            else:
                await update.message.reply_text('No img available yet...')
        return _send_live_pic_cmd_callback

    def _get_send_last_casc_pic_cmd_callback(self) -> TelegramCallbackType:
        bot = self

        async def _send_last_casc_pic_cmd_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            nonlocal bot
            if bot.node_last_casc_img is not None:
                caption = 'Last Cascade:'
                await update.message.reply_photo(bot.node_last_casc_img, caption)
            else:
                await update.message.reply_text('No casc img available yet...')
        return _send_last_casc_pic_cmd_callback

    def _get_let_in_callback(self) -> TelegramCallbackType:
        bot = self
        seconds = flap_config.let_in_open_seconds

        async def _let_in_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            if bot.is_ongoing_let_in:
                await update.message.reply_text(f"Oops... There is already a 'letin' command in execution. Ignoring...")
                return

            bot.is_ongoing_let_in = True
            await update.message.reply_text(f"Ok, door is open for {seconds}s...")
            await bot.flap_handler.unlock_flap_for_let_in()
            await asyncio.sleep(seconds)
            await update.message.reply_text(f"Locking door after {seconds}s...")
            await bot.flap_handler.finish_letin()
            bot.is_ongoing_let_in = False
        return _let_in_callback

    def _get_cancel_let_in_callback(self) -> TelegramCallbackType:
        bot = self

        async def _cancel_let_in_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            nonlocal bot
            if bot.is_ongoing_let_in:
                await update.message.reply_text(f"Cancelling last 'letin' command")
                bot.is_ongoing_let_in = False
                await bot.flap_handler.finish_letin()
            else:
                await update.message.reply_text(f"No 'letin' command to cancel")
        return _cancel_let_in_callback

    def _get_lock_moria_callback(self) -> TelegramCallbackType:
        bot = self

        async def _lock_moria(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            nonlocal bot
            await update.message.reply_text("Locking Moria...")
            message = await bot.flap_handler.lock_moria()
            await update.message.reply_text(message)
        return _lock_moria

    def _get_lock_moria_in_callback(self) -> TelegramCallbackType:
        bot = self

        async def _lock_moria_in(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            nonlocal bot
            message = await bot.flap_handler.lock_moria_in()
            await update.message.reply_text(message)
        return _lock_moria_in

    def _get_lock_moria_out_callback(self) -> TelegramCallbackType:
        bot = self

        async def _lock_moria_out(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            nonlocal bot
            message = await bot.flap_handler.lock_moria_out()
            await update.message.reply_text(message)
        return _lock_moria_out

    def _get_unlock_moria_callback(self) -> TelegramCallbackType:
        bot = self

        async def _unlock_moria(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            nonlocal bot
            message = await bot.flap_handler.unlock_moria()
            await update.message.reply_text(message)
        return _unlock_moria

    def _get_status_pets_callback(self) -> TelegramCallbackType:
        bot = self

        async def _send_pets_data_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            nonlocal bot
            message = await bot.flap_handler.get_pets_status_str()
            await update.message.reply_text(message)
        return _send_pets_data_callback

    def _get_mute_notifications_callback(self) -> TelegramCallbackType:
        bot = self

        async def _mute_notifications(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            nonlocal bot
            # Util function used to mute the sending of verdicts
            timeout = general_config.mute_img_send_minutes
            await update.message.reply_text(f"Muting Balrog image notifications for the next {timeout} minutes")
            bot.muted_images = True
            await asyncio.sleep(60 * timeout)
            bot.muted_images = False
            await update.message.reply_text("Restarting Balrog image notifications")
        return _mute_notifications

    def _get_switch_pet_location_callback(self, pet_id: int) -> TelegramCallbackType:
        bot = self

        async def _switch_pet_location_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            nonlocal bot, pet_id
            message = await bot.flap_handler.switch_pet_location(pet_id)
            await update.message.reply_text(message)
        return _switch_pet_location_callback

    def _get_send_device_data_callback(self, device_id: int) -> TelegramCallbackType:
        bot = self

        async def _send_device_data_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            nonlocal bot, device_id
            message = await bot.flap_handler.send_device_data(device_id)
            await update.message.reply_text(message)
        return _send_device_data_callback

    def _get_activate_curfew_callback(self) -> TelegramCallbackType:
        bot = self

        async def _activate_curfew_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            nonlocal bot
            message = await bot.flap_handler.activate_curfew()
            await update.message.reply_text(message)
        return _activate_curfew_callback


class DebugBot(MessageSender):
    def __init__(self):
        super().__init__()

    def send_img(self, img: MatLike, caption: str) -> None:
        # Nothing to do here; we simply ignore the invocation
        logger.warning(f"DebugTelegramBot - Ignoring sending image!")

    def send_text(self, message: str) -> None:
        # Nothing to do here; we simply ignore the invocation
        logger.warning(f"DebugTelegramBot - Ignoring sending text!")
