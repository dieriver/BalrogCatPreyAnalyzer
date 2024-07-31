import asyncio
import os
import time
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from typing import Any, Callable, Dict, Coroutine, Optional

import cv2
from telegram import Update, Message, Bot
from telegram.ext import Application, CommandHandler, ContextTypes

from balrog.config import flap_config, general_config, command_aliases_config
from balrog.interface import MessageSender
from balrog.interface.flap_locker import FlapLocker
from balrog.utils import Logging, logger

_TelegramCallbackType = Callable[[Update, ContextTypes.DEFAULT_TYPE], Coroutine[Any, Any, None]]


class _LockMode(Enum):
    FULL = auto()
    LOCK_IN = auto()
    LOCK_OUT = auto()
    UNLOCK = auto()
    CURFEW = auto()


class BalrogTelegramBot(MessageSender):
    def __init__(self):
        # Insert Chat ID and Bot Token according to Telegram API
        super().__init__()
        if os.getenv('TELEGRAM_CHAT_ID') == "":
            raise Exception("Telegram CHAT ID not set!. Please set the 'TELEGRAM_CHAT_ID' environment variable")
        if os.getenv('TELEGRAM_BOT_TOKEN') == "":
            raise Exception("Telegram Bot token not set!. Please set the 'TELEGRAM_BOT_TOKEN' environment variable")
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        app_builder = Application.builder()
        app_builder.token(self.bot_token)
        app_builder.concurrent_updates(True)
        app_builder.post_init(self.get_send_message("Balrog raises from the abyss..."))
        app_builder.post_stop(self.get_send_message("Balrog goes back to the abyss... for now..."))
        self.telegram_endpoint = app_builder.build()
        self.flap_handler = FlapLocker()
        self.commands: Dict[str, _TelegramCallbackType] = dict()
        self.message_loop = asyncio.get_event_loop()
        pets_data = self.message_loop.run_until_complete(self.flap_handler.get_pets_data())
        devices_data = self.message_loop.run_until_complete(self.flap_handler.get_devices_data())

        self._populate_supported_commands(pets_data, devices_data)
        self._populate_command_aliases()
        self._is_ongoing_let_in: bool = False
        self.mute_msg: Optional[Message] = None

        # Add all commands to handler
        handlers = []
        for command in self.commands:
            logger.info(f"Registering command '{command}'")
            handlers.append(CommandHandler(command, self.commands[command]))
        self.telegram_endpoint.add_handlers(handlers)

    @property
    def is_ongoing_let_in(self) -> bool:
        return self._is_ongoing_let_in

    @is_ongoing_let_in.setter
    def is_ongoing_let_in(self, new_val: bool) -> None:
        self._is_ongoing_let_in = new_val

    def _populate_command_aliases(self):
        for command in command_aliases_config.aliases_map:
            for alias in command_aliases_config.aliases_map[command]:
                self.commands[alias] = self.commands[command]

    # Constructor supporter functions
    def _populate_supported_commands(self, pets_data: Dict[str, int], devices_data: Dict[str, int]) -> None:
        self.commands['help'] = self._get_help_cmd_callback()
        self.commands['clean'] = self._get_clean_cmd_callback()
        self.commands['restart'] = self._get_restart_cmd_callback()
        self.commands['statusBalrog'] = self._get_node_status_cmd_callback()
        self.commands['sendlivepic'] = self._get_send_live_pic_cmd_callback()
        self.commands['sendlastcascpic'] = self._get_send_last_casc_pic_cmd_callback()
        self.commands['letin'] = self._get_let_in_callback()
        self.commands['cancelLetin'] = self._get_cancel_let_in_callback()
        self.commands['lock'] = self._get_lock_moria_callback_for_status(_LockMode.FULL)
        self.commands['lockin'] = self._get_lock_moria_callback_for_status(_LockMode.LOCK_IN)
        self.commands['lockout'] = self._get_lock_moria_callback_for_status(_LockMode.LOCK_OUT)
        self.commands['unlock'] = self._get_lock_moria_callback_for_status(_LockMode.UNLOCK)
        self.commands['statusPets'] = self._get_status_pets_callback()
        self.commands['mute'] = self._get_mute_notifications_callback()
        self.commands['unmute'] = self._get_resume_notifications_callback()
        # create callbacks for switching the state of pets
        for name, pet_id in pets_data.items():
            self.commands[f'switch{name}'] = self._get_switch_pet_location_callback(pet_id)
        # create callbacks for status of the devices
        for name, device_id in devices_data.items():
            self.commands[f'status{name}'] = self._get_send_device_data_callback(device_id)
        # Not very used commands
        self.commands['curfew'] = self._get_lock_moria_callback_for_status(_LockMode.CURFEW)

    # Telegram thread supporter functions
    def start(self) -> None:
        # Start the polling stuff. this locks the current thread
        self.telegram_endpoint.run_polling(allowed_updates=[Update.MESSAGE], stop_signals=[])
        # We wait for the
        time.sleep(2)

    def stop(self) -> None:
        self.telegram_endpoint.stop_running()

    def get_send_message(self, msg: str) -> Callable[[Application], Coroutine[Any, Any, None]]:
        chat_id = self.chat_id

        async def _send_hello_message(app: Application) -> None:
            nonlocal chat_id, msg
            await app.bot.send_message(chat_id=chat_id, text=msg)
        return _send_hello_message

    # Raw send text and img functions

    def send_text(self, message: str) -> None:
        data = {
            "msg": message,
            "chat_id": self.chat_id
        }

        async def _send_text_callback(args: Dict[str, Any], init_tstamp: datetime, bot: Bot) -> None:
            sent_tstamp = datetime.now()
            await bot.send_message(
                chat_id=args["chat_id"],
                text=args["msg"]
            )
            delta = (sent_tstamp - init_tstamp).seconds
            logger.info(f"Sender - Msg: '{args['msg']}', Sent: {sent_tstamp}, Total: {delta:.2f}s")
        scheduled_tstamp = datetime.now()
        logger.info(f"Sender - Msg: '{message}', Scheduled: {scheduled_tstamp}")
        self.telegram_endpoint.create_task(_send_text_callback(data, scheduled_tstamp, self.telegram_endpoint.bot))

    def send_img(self, img: Path, caption: str, force_send: bool = False) -> None:
        data = {
            "caption": caption,
            "img_path": str(img),
            "force_send": force_send,
            "muted_images": self.muted_images,
            "chat_id": self.chat_id
        }

        async def _send_img_callback(args: Dict[str, Any], init_tstamp: datetime, bot: Bot) -> None:
            sent_tstamp = datetime.now()
            try:
                if not args["force_send"] and args["muted_images"]:
                    return
                await bot.send_photo(
                    chat_id=args["chat_id"],
                    photo=open(args["img_path"], 'rb'),
                    caption=args["caption"]
                )
                delta = (sent_tstamp - init_tstamp).seconds
                logger.info(f"Sender - File: {data['img_path']}, Sent: {sent_tstamp}, Total: {delta:.2f}s")
            finally:
                os.remove(data["img_path"])
        scheduled_tstamp = datetime.now()
        logger.info(f"Sender - File: {str(img)}, Scheduled: {scheduled_tstamp}")
        self.telegram_endpoint.create_task(_send_img_callback(data, scheduled_tstamp, self.telegram_endpoint.bot))

    def _get_help_cmd_callback(self) -> _TelegramCallbackType:
        bot = self

        async def help_cmd_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            nonlocal bot
            bot_message = 'Following commands supported:'
            for command in bot.commands:
                bot_message += '\n /' + command
            await update.message.reply_text(bot_message)
        return help_cmd_callback

    def _get_clean_cmd_callback(self) -> _TelegramCallbackType:
        async def clean_cmd_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            new_msg = await update.message.reply_text('Cleaning old logs...')
            removed_paths = Logging.clean_logs()
            await new_msg.reply_text(f'Removed: [{*removed_paths,}]')
        return clean_cmd_callback

    def _get_restart_cmd_callback(self) -> _TelegramCallbackType:
        bot = self

        async def _restart_cmd_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            nonlocal bot
            await update.message.reply_text('Restarting script...')
            bot.stop()
        return _restart_cmd_callback

    def _get_node_status_cmd_callback(self) -> _TelegramCallbackType:
        bot = self

        async def _node_status_cmd_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            nonlocal bot
            rdy_for_img = str(bot.frames_rdy_for_img) if bot.frames_rdy_for_img is not None else "Unknown"
            rdy_for_casc = str(bot.frames_rdy_for_cascade) if bot.frames_rdy_for_cascade is not None else "Unknown"
            rdy_for_agg = str(bot.frames_rdy_for_aggregate) if bot.frames_rdy_for_aggregate is not None else "Unknown"
            last_casc_time = str(bot.last_casc_time) if bot.last_casc_time is not None else "Unknown"
            roundtrip_delay = str(bot.queue_avg_delay) if bot.queue_avg_delay is not None else "Unknown"

            bot_message = (f'Frames rdy for image: {rdy_for_img}\n'
                           f'Frames rdy for cascade: {rdy_for_casc}\n'
                           f'Frames rdy for aggregation: {rdy_for_agg}\n'
                           f'Last cascade time: {last_casc_time}s\n'
                           f'Frame roundtrip delay: {roundtrip_delay}s\n'
                           f"Notifications: {'Disabled' if bot.muted_images else 'Enabled'}")
            await update.message.reply_text(bot_message)
        return _node_status_cmd_callback

    def _get_send_live_pic_cmd_callback(self) -> _TelegramCallbackType:
        bot = self

        async def _send_live_pic_cmd_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            nonlocal bot
            if bot.live_img is not None:
                caption = 'Here it is...'
                with TemporaryDirectory() as tmp_dir:
                    cv2.imwrite(f'{tmp_dir}/balrog_send_live_img.jpg', bot.live_img)
                    await update.message.reply_photo(f'{tmp_dir}/balrog_send_live_img.jpg', caption)
            else:
                await update.message.reply_text('No img available yet...')
        return _send_live_pic_cmd_callback

    def _get_send_last_casc_pic_cmd_callback(self) -> _TelegramCallbackType:
        bot = self

        async def _send_last_casc_pic_cmd_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            nonlocal bot
            if bot.last_casc_img is not None:
                caption = 'Last Cascade:'
                with TemporaryDirectory() as tmp_dir:
                    cv2.imwrite(f'{tmp_dir}/balrog_send_casc_img.jpg', bot.last_casc_img)
                    await update.message.reply_photo(f'{tmp_dir}/balrog_send_casc_img.jpg', caption)
            else:
                await update.message.reply_text('No casc img available yet...')
        return _send_last_casc_pic_cmd_callback

    def _get_let_in_callback(self) -> _TelegramCallbackType:
        bot = self
        seconds = flap_config.let_in_open_seconds

        async def _let_in_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            nonlocal bot
            if bot.is_ongoing_let_in:
                await update.message.reply_text(f"Oops... There is already a 'letin' command in execution. Ignoring...")
                return

            bot.is_ongoing_let_in = True
            open_msg = await update.message.reply_text(f"Ok, door is open for {seconds}s...")
            result = await bot.flap_handler.unlock_flap_for_let_in()
            await open_msg.reply_text(result)

            async def _finish_let_in(ctx: ContextTypes.DEFAULT_TYPE) -> None:
                nonlocal bot, update
                lock_msg = await update.message.reply_text(f"Locking door after {seconds}s...")
                result_lock = await bot.flap_handler.finish_letin()
                await lock_msg.reply_text(result_lock)
                bot.is_ongoing_let_in = False

            context.job_queue.run_once(_finish_let_in, seconds)
        return _let_in_callback

    def _get_cancel_let_in_callback(self) -> _TelegramCallbackType:
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

    def _get_lock_moria_callback_for_status(self, mode: _LockMode):
        match mode:
            case _LockMode.FULL:
                message = "Locking Moria fully..."
                callback = self.flap_handler.lock_moria
            case _LockMode.LOCK_IN:
                message = "Locking Moria for outgoing..."
                callback = self.flap_handler.lock_moria_in
            case _LockMode.LOCK_OUT:
                message = "Locking Moria for incoming..."
                callback = self.flap_handler.lock_moria_out
            case _LockMode.UNLOCK:
                message = "Unlocking Moria..."
                callback = self.flap_handler.unlock_moria
            case _LockMode.CURFEW:
                message = "Activating curfew on Moria..."
                callback = self.flap_handler.activate_curfew
            case _:
                raise RuntimeError(f"Unhandled case for locking mode '{mode}'")

        async def _lock_moria(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            nonlocal message, callback
            lock_msg = await update.message.reply_text(message)
            lock_result = await callback()
            await lock_msg.reply_text(lock_result)
        return _lock_moria

    def _get_status_pets_callback(self) -> _TelegramCallbackType:
        bot = self

        async def _send_pets_data_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            nonlocal bot
            message = await bot.flap_handler.get_pets_status_str()
            await update.message.reply_text(message)
        return _send_pets_data_callback

    @staticmethod
    async def _resume_notifications(ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not ctx.job.data.muted_images or ctx.job.data.mute_msg is None:
            ctx.job.data.send_text("Images were not muted; Ignoring.")

        for job in ctx.job_queue.get_jobs_by_name("resume_notifications"):
            job.schedule_removal()

        ctx.job.data.muted_images = False
        await ctx.job.data.mute_msg.reply_text("Restarting Balrog image notifications")
        ctx.job.data.mute_msg = None

    def _get_mute_notifications_callback(self) -> _TelegramCallbackType:
        bot = self
        timeout = general_config.mute_img_send_minutes

        async def _mute_notifications(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            nonlocal bot, timeout
            if bot.muted_images:
                await update.message.reply_text(f"Image notifications were already muted; Ignoring.")
                return

            delay = timeout
            try:
                if context.args is not None and len(context.args) >= 1:
                    delay = int(context.args[0])
            except ValueError:
                # Nothing to do here; we use the default delay if we couldn't parse the first argument
                pass

            # Util function used to mute the sending of verdicts
            bot.mute_msg = await update.message.reply_text(f"Muting Balrog image notifications "
                                                           f"for the next {delay} minutes")
            bot.muted_images = True
            context.job_queue.run_once(
                BalrogTelegramBot._resume_notifications,
                60 * delay,
                data=bot,
                name="resume_notifications"
            )
        return _mute_notifications

    def _get_resume_notifications_callback(self) -> _TelegramCallbackType:
        bot = self

        async def _resume_notifications(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            nonlocal bot
            bot.mute_msg = update.message
            bot.unmute_job = context.job_queue.run_once(
                BalrogTelegramBot._resume_notifications,
                0,
                data=bot
            )
        return _resume_notifications

    def _get_switch_pet_location_callback(self, pet_id: int) -> _TelegramCallbackType:
        bot = self

        async def _switch_pet_location_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            nonlocal bot, pet_id
            switch_result = await bot.flap_handler.switch_pet_location(pet_id)
            await update.message.reply_text(switch_result)
        return _switch_pet_location_callback

    def _get_send_device_data_callback(self, device_id: int) -> _TelegramCallbackType:
        bot = self

        async def _send_device_data_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            nonlocal bot, device_id
            device_result = await bot.flap_handler.get_device_data_str(device_id)
            await update.message.reply_text(device_result)
        return _send_device_data_callback


class DebugBot(MessageSender):
    def __init__(self):
        super().__init__()
        self.stop_event = Event()

    def start(self) -> None:
        while not self.stop_event.is_set():
            time.sleep(1)

    def stop(self) -> None:
        self.stop_event.set()

    def send_img(self, img: Path, caption: str) -> None:
        # Nothing to do here; we simply ignore the invocation
        logger.warning(f"DebugTelegramBot - Ignoring sending image: {str(img)}! - Caption: '{caption}'")

    def send_text(self, message: str) -> None:
        # Nothing to do here; we simply ignore the invocation
        logger.warning(f"DebugTelegramBot - Ignoring sending text! - Caption: '{message}'")
