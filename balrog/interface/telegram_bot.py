import asyncio
import os
import time
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from typing import Any, Callable, Dict, Coroutine, Optional, List, Self

import cv2
from telegram import Update, Message
from telegram.constants import ReactionEmoji
from telegram.ext import Application, CommandHandler, ContextTypes, CallbackContext

from balrog.config import flap_config, general_config, command_aliases_config
from balrog.interface import MessageSender
from balrog.interface.flap_locker import FlapLocker
from balrog.utils import Logging, logger

_TelegramCmdCallbackType = Callable[[Update, ContextTypes.DEFAULT_TYPE], Coroutine[Any, Any, None]]
_SchedFuncCallbackType = Callable[[ContextTypes.DEFAULT_TYPE], Coroutine[Any, Any, None]]


async def _get_value_from_var_or_args(var: Optional[str], args: Optional[List[str]],
                                      update: Update, error_msg: str) -> Optional[str]:
    if var is None:
        if args is not None and len(args) >= 1:
            return args[0]
        else:
            await update.message.reply_text(error_msg)
            return None
    else:
        return var


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
        self.telegram_endpoint: Application = app_builder.build()
        self.flap_handler: FlapLocker = FlapLocker()
        self.commands: Dict[str, _TelegramCmdCallbackType] = dict()
        self.message_loop = asyncio.get_event_loop()
        self.pets_data: Dict[str, int] = self.message_loop.run_until_complete(self.flap_handler.get_pets_data())
        self.devices_data: Dict[str, int] = self.message_loop.run_until_complete(self.flap_handler.get_devices_data())

        self._populate_supported_commands()
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
    def default_flap_device(self) -> str:
        return flap_config.default_flap

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
    def _populate_supported_commands(self) -> None:
        self.commands['help'] = self._get_help_cmd_callback()
        self.commands['clean'] = self._get_clean_cmd_callback()
        self.commands['restart'] = self._get_restart_cmd_callback()
        self.commands['sendlivepic'] = self._get_send_live_pic_cmd_callback()
        self.commands['sendlastcascpic'] = self._get_send_last_casc_pic_cmd_callback()
        self.commands['letin'] = self._get_let_in_callback(device_name=self.default_flap_device)
        self.commands['cancelLetin'] = self._get_cancel_let_in_callback(device_name=self.default_flap_device)
        self.commands['lock'] = self._get_lock_moria_callback_for_status(device_name=self.default_flap_device,
                                                                         mode=_LockMode.FULL)
        self.commands['lockin'] = self._get_lock_moria_callback_for_status(device_name=self.default_flap_device,
                                                                           mode=_LockMode.LOCK_IN)
        self.commands['lockout'] = self._get_lock_moria_callback_for_status(device_name=self.default_flap_device,
                                                                            mode=_LockMode.LOCK_OUT)
        self.commands['unlock'] = self._get_lock_moria_callback_for_status(device_name=self.default_flap_device,
                                                                           mode=_LockMode.UNLOCK)
        self.commands['mute'] = self._get_mute_notifications_callback()
        self.commands['unmute'] = self._get_resume_notifications_callback()
        self.commands['switch'] = self._get_switch_location_callback()
        for pet_name in self.pets_data:
            # Callbacks for switching the state of pets
            self.commands[f'switch{pet_name}'] = self._get_switch_location_callback(pet_name=pet_name)
            # Callbacks for getting the state of pets
            self.commands[f'status{pet_name}'] = self._get_status_callback(status_arg=pet_name)
        self.commands['status'] = self._get_status_callback()
        self.commands['statusPets'] = self._get_status_callback(status_arg="pets")
        self.commands['statusBalrog'] = self._get_status_callback(status_arg="balrog")
        # create callbacks for status of the devices
        for device_name in self.devices_data:
            self.commands[f'status{device_name}'] = self._get_status_callback(status_arg=device_name)

        # Not very used commands
        self.commands['curfew'] = self._get_lock_moria_callback_for_status(device_name=self.default_flap_device,
                                                                           mode=_LockMode.CURFEW)

    # Telegram thread supporter functions
    def start(self) -> None:
        # Start the polling stuff. this locks the current thread
        self.telegram_endpoint.run_polling(allowed_updates=[Update.MESSAGE], stop_signals=[])
        # We wait for the goodbye message to be sent
        # time.sleep(2)

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
        scheduled_tstamp = datetime.now()
        data = {
            "msg": message,
            "init_tstamp": scheduled_tstamp
        }

        async def _send_text_callback(context: CallbackContext) -> None:
            sent_tstamp = datetime.now()
            await context.bot.send_message(
                chat_id=context.job.chat_id,
                text=context.job.data["msg"]
            )
            delta = (sent_tstamp - context.job.data["init_tstamp"]).seconds
            logger.info(f"Sender - Msg: '{context.job.data['msg']}', Sent: {sent_tstamp}, Total: {delta:.2f}s")
        logger.info(f"Sender - Msg: '{message}', Scheduled: {scheduled_tstamp}")
        self.telegram_endpoint.job_queue.run_once(_send_text_callback, 0, data=data, chat_id=self.chat_id)

    def send_img(self, img: Path, caption: str, force_send: bool = False) -> None:
        scheduled_tstamp = datetime.now()
        data = {
            "caption": caption,
            "img_path": str(img),
            "force_send": force_send,
            "muted_images": self.muted_images,
            "init_tstamp": scheduled_tstamp
        }

        async def _send_img_callback(context: CallbackContext) -> None:
            sent_tstamp = datetime.now()
            try:
                if not context.job.data["force_send"] and context.job.data["muted_images"]:
                    return
                await context.bot.send_photo(
                    chat_id=context.job.chat_id,
                    photo=open(context.job.data["img_path"], 'rb'),
                    caption=context.job.data["caption"]
                )
                delta = (sent_tstamp - context.job.data["init_tstamp"]).seconds
                logger.info(f"Sender - File: {data['img_path']}, Sent: {sent_tstamp}, Total: {delta:.2f}s")
            finally:
                os.remove(data["img_path"])
        logger.info(f"Sender - File: {str(img)}, Scheduled: {scheduled_tstamp}")
        self.telegram_endpoint.job_queue.run_once(_send_img_callback, 0, data=data, chat_id=self.chat_id)

    def _get_help_cmd_callback(self) -> _TelegramCmdCallbackType:
        bot = self

        async def help_cmd_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            nonlocal bot
            bot_message = 'Following commands supported:'
            for command in bot.commands:
                bot_message += '\n /' + command
            await update.message.reply_text(bot_message)
        return help_cmd_callback

    def _get_clean_cmd_callback(self) -> _TelegramCmdCallbackType:
        async def clean_cmd_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            new_msg = await update.message.reply_text('Cleaning old logs...')
            removed_paths = Logging.clean_logs()
            await new_msg.reply_text(f'Removed: [{*removed_paths,}]')
        return clean_cmd_callback

    def _get_restart_cmd_callback(self) -> _TelegramCmdCallbackType:
        bot = self

        async def _restart_cmd_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            nonlocal bot
            await update.message.reply_text('Restarting script...')
            bot.stop()
        return _restart_cmd_callback

    def _get_balrog_status_cmd_callback(self) -> _TelegramCmdCallbackType:
        bot = self

        async def _node_status_cmd_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            nonlocal bot
            rdy_for_img = f"{bot.frames_rdy_for_img}" if bot.frames_rdy_for_img is not None else "Unknown"
            rdy_for_casc = f"{bot.frames_rdy_for_cascade}" if bot.frames_rdy_for_cascade is not None else "Unknown"
            rdy_for_agg = f"{bot.frames_rdy_for_aggregate}" if bot.frames_rdy_for_aggregate is not None else "Unknown"
            last_casc_time = f"{bot.last_casc_time} s." if bot.last_casc_time is not None else "Unknown"
            roundtrip_delay = f"{bot.queue_avg_delay} s." if bot.queue_avg_delay is not None else "Unknown"

            bot_message = (f'Frames rdy for image: {rdy_for_img}\n'
                           f'Frames rdy for cascade: {rdy_for_casc}\n'
                           f'Frames rdy for aggregation: {rdy_for_agg}\n'
                           f'Last cascade time: {last_casc_time}\n'
                           f'Frame roundtrip delay: {roundtrip_delay}\n'
                           f"Notifications: {'Disabled' if bot.muted_images else 'Enabled'}")
            await update.message.reply_text(bot_message)
        return _node_status_cmd_callback

    def _get_send_live_pic_cmd_callback(self) -> _TelegramCmdCallbackType:
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

    def _get_send_last_casc_pic_cmd_callback(self) -> _TelegramCmdCallbackType:
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

    def _get_report_flap_command_timeout(self, command: str, device_id: int, message: Message) -> _SchedFuncCallbackType:
        bot = self
        async def _report_flap_command_timeout(context: ContextTypes.DEFAULT_TYPE) -> None:
            nonlocal bot, command, device_id, message
            await message.reply_text(f"The last '{command}' command was not acknowledged on time:\n"
                                     "Please check the current status of the door:")
            flap_data = await bot.flap_handler.get_device_data_str(device_id)
            await context.bot.send_message(bot.chat_id, flap_data)
        return _report_flap_command_timeout

    def _get_finish_let_in_callback(self, update: Update, device_id: int, seconds: int) -> _SchedFuncCallbackType:
        bot = self
        async def _finish_let_in(context: ContextTypes.DEFAULT_TYPE) -> None:
            nonlocal bot, update, seconds, device_id
            lock_msg = await update.message.reply_text(f"Locking door after {seconds}s...")
            result_success = await bot.flap_handler.finish_letin(device_id)
            react = ReactionEmoji.THUMBS_UP if result_success else ReactionEmoji.THUMBS_DOWN
            await lock_msg.set_reaction(react)
            bot.is_ongoing_let_in = False

            if not result_success:
                context.job_queue.run_once(
                    bot._get_report_flap_command_timeout("lock", device_id, lock_msg),
                    flap_config.seconds_to_wait_before_reporting_timeout
                )
        return _finish_let_in

    def _get_let_in_callback(self, device_name: str) -> _TelegramCmdCallbackType:
        bot = self
        seconds = flap_config.let_in_open_seconds
        flap_id = self.devices_data[device_name.lower()]

        async def _let_in_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            nonlocal bot, seconds, flap_id
            if bot.is_ongoing_let_in:
                await update.message.reply_text(f"Oops... There is already a 'letin' command in execution. Ignoring...")
                return

            bot.is_ongoing_let_in = True
            open_msg = await update.message.reply_text(f"Ok, door is open for {seconds}s...")
            let_in_success = await bot.flap_handler.unlock_flap_for_let_in(flap_id)
            reaction = ReactionEmoji.THUMBS_UP if let_in_success else ReactionEmoji.THUMBS_DOWN
            await open_msg.set_reaction(reaction)

            if not let_in_success:
                bot.is_ongoing_let_in = False
                context.job_queue.run_once(
                    bot._get_report_flap_command_timeout("unlock", flap_id, open_msg),
                    flap_config.seconds_to_wait_before_reporting_timeout
                )
                return

            context.job_queue.run_once(self._get_finish_let_in_callback(update, flap_id, seconds), seconds)
        return _let_in_callback

    def _get_cancel_let_in_callback(self, device_name: str) -> _TelegramCmdCallbackType:
        bot = self
        flap_id = self.devices_data[device_name.lower()]

        async def _cancel_let_in_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            nonlocal bot, flap_id
            if bot.is_ongoing_let_in:
                await update.message.reply_text(f"Cancelling last 'letin' command")
                bot.is_ongoing_let_in = False
                await bot.flap_handler.finish_letin(flap_id)
            else:
                await update.message.reply_text(f"No 'letin' command to cancel")
        return _cancel_let_in_callback

    def _get_lock_moria_callback_for_status(self, device_name: str, mode: _LockMode):
        bot = self
        device_id = self.devices_data[device_name.lower()]

        async def _set_device_lock_state(update: Update, contex: ContextTypes.DEFAULT_TYPE) -> None:
            nonlocal bot, mode, device_id
            match mode:
                case _LockMode.FULL:
                    message = f"Locking {device_name} fully..."
                    command = "lock"
                    callback = bot.flap_handler.device_lock
                case _LockMode.LOCK_IN:
                    message = f"Locking {device_name} for outgoing..."
                    command = "lock in"
                    callback = bot.flap_handler.device_lock_in
                case _LockMode.LOCK_OUT:
                    message = f"Locking {device_name} for incoming..."
                    command = "lock out"
                    callback = bot.flap_handler.device_lock_out
                case _LockMode.UNLOCK:
                    message = f"Unlocking {device_name}..."
                    command = "unlock"
                    callback = bot.flap_handler.unlock_device
                case _LockMode.CURFEW:
                    message = f"Activating curfew on {device_name}..."
                    command = "curfew"
                    callback = bot.flap_handler.device_curfew
                case _:
                    raise RuntimeError(f"Unhandled case for locking mode '{mode}' on '{device_name}")
            lock_msg = await update.message.reply_text(message)
            lock_success = await callback(device_id)
            reaction = ReactionEmoji.THUMBS_UP if lock_success else ReactionEmoji.THUMBS_DOWN
            await lock_msg.set_reaction(reaction)

            if not lock_success:
                contex.job_queue.run_once(
                    bot._get_report_flap_command_timeout(command, device_id, lock_msg),
                    flap_config.seconds_to_wait_before_reporting_timeout
                )
        return _set_device_lock_state

    def _get_status_all_pets_callback(self) -> _TelegramCmdCallbackType:
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
            return

        for job in ctx.job_queue.get_jobs_by_name("resume_notifications"):
            job.schedule_removal()

        ctx.job.data.muted_images = False
        await ctx.job.data.mute_msg.reply_text("Restarting Balrog image notifications")
        ctx.job.data.mute_msg = None

    def _get_mute_notifications_callback(self) -> _TelegramCmdCallbackType:
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

    def _get_resume_notifications_callback(self) -> _TelegramCmdCallbackType:
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

    def _get_switch_location_callback(self, pet_name: Optional[str] = None) -> _TelegramCmdCallbackType:
        bot = self

        async def _switch_location_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            nonlocal bot, pet_name

            pet = await _get_value_from_var_or_args(pet_name, context.args, update, "Which pet??...")
            if pet is None:
                return

            if pet.lower() not in bot.pets_data:
                await update.message.reply_text(f"Pet '{pet}' is unknown...")
                return
            pet_id = bot.pets_data[pet.lower()]

            switch_result = await bot.flap_handler.switch_pet_location(pet_id)
            await update.message.reply_text(switch_result)
        return _switch_location_callback

    def _get_status_callback(self, status_arg: Optional[str] = None) -> _TelegramCmdCallbackType:
        bot = self

        async def _send_device_data_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            nonlocal bot, status_arg

            arg = await _get_value_from_var_or_args(status_arg, context.args, update, "Status of what??...")
            if arg is None:
                return

            arg_low = arg.lower()

            match arg_low:
                case "balrog":
                    balrog_status_coro = self._get_balrog_status_cmd_callback()
                    return await balrog_status_coro(update, context)
                case "pets":
                    pets_status_coro = self._get_status_all_pets_callback()
                    return await pets_status_coro(update, context)
                case arg_low if arg_low in bot.devices_data:
                    device_id = bot.devices_data[arg_low]
                    device_result = await bot.flap_handler.get_device_data_str(device_id)
                    await update.message.reply_text(device_result)
                case arg_low if arg_low in bot.pets_data:
                    pet_id = bot.pets_data[arg_low]
                    pet_result = await bot.flap_handler.get_pets_status_str(filter_by_id=pet_id)
                    await update.message.reply_text(pet_result)
                case _:
                    await update.message.reply_text(f"Device or pet '{arg}' is unknown...")
                    return
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
