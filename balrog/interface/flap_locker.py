import asyncio
import os
from datetime import datetime
from typing import Any, Dict, List

import pytz
from surepy import Surepy, SurepyEntity, SurepyDevice, EntityType
from surepy.entities.devices import Flap
from surepy.entities.pet import Pet
from surepy.enums import LockState, Location

from balrog.config import general_config
from balrog.interface import MessageSender
from balrog.utils.utils import logger


class FlapLocker:
    def __init__(self):
        # user/password authentication (gets a token in background)
        if os.getenv('SUREPET_USER') == "":
            raise Exception("Surepet username not set!. Please set the 'SUREPET_USER' environment variable")
        if os.getenv('SUREPET_PASSWORD') == "":
            raise Exception("Surepet password not set!. Please set the 'SUREPET_PASSWORD' environment variable")
        self.surepy = Surepy(email=os.getenv('SUREPET_USER'), password=os.getenv('SUREPET_PASSWORD'))

    # Functions used to "introspect" the information about pets and devices
    # to register commands
    async def get_pets_data(self, *args) -> Dict[str, int]:
        registered_pets: List[Pet] = await self.surepy.get_pets()
        pets_data: Dict[str, int] = dict()
        for registered_pet in registered_pets:
            pets_data[registered_pet.name] = registered_pet.pet_id
        return pets_data

    async def get_devices_data(self, *args) -> Dict[str, int]:
        registered_devices: List[SurepyDevice] = await self._get_fresh_devices()
        devices_data: Dict[str, int] = dict()
        for registered_device in registered_devices:
            devices_data[registered_device.name] = registered_device.id
        return devices_data

    # Functions used to send data from surepy to the telegram interface
    async def send_pets_data(self, msg_sender: MessageSender, _: Any) -> None:
        # list with all pets
        pets: List[Dict[str, Any]] = await self.surepy.sac.get_pets()
        message = f"I found this:"
        for pet in pets:
            location: Location = Location(pet['status']['activity']['where'])
            location_since: datetime = datetime.fromisoformat(pet['status']['activity']['since'])
            corrected_since: datetime = location_since.astimezone(pytz.timezone(general_config.local_timezone))
            message += (f"\nPet '{pet['name']}', location: {location}, "
                        f"since: {corrected_since.strftime(general_config.timestamp_format)}")
        msg_sender.send_text(message)

    async def send_device_data(self, msg_sender: MessageSender, device_id: int) -> None:
        devices: List[SurepyDevice] = await self._get_fresh_devices()
        for device in devices:
            if device.id == device_id:
                if isinstance(device, Flap):
                    lock_status = device.state
                else:
                    lock_status = LockState.UNLOCKED
                lock_status_str = str(lock_status).replace('_', ' ')
                msg_sender.send_text(f"I found this:\n"
                                       f"Device: '{device.name}', "
                                       f"Lock State: '{lock_status_str}', "
                                       f"Battery Level: '{device.battery_level}'")
                return
        msg_sender.send_text(f"I could not find the device")

    async def list_devices(self, msg_sender: MessageSender) -> None:
        # all entities as id-indexed dict
        entities: Dict[int, SurepyEntity] = await self.surepy.get_entities()

        # list with all devices
        devices: List[SurepyDevice] = await self._get_fresh_devices()
        for device in devices:
            msg_sender.send_text(f"{device.name = } | {device.serial = } | {device.battery_level = }")
            msg_sender.send_text(f"{device.type = } | {device.unique_id = } | {device.id = }")
            msg_sender.send_text(f"{entities[device.parent_id].full_name = } | {entities[device.parent_id] = }\n")

    async def get_lock_state(self) -> LockState:
        try:
            devices: List[SurepyDevice] = await self._get_fresh_devices()
            for device in devices:
                if device.type == EntityType.CAT_FLAP:
                    cat_flap: Flap = device
                    return cat_flap.state
        except Exception:
            logger.exception('+++ Exception while getting last flap state: ')
            # We assume a default value;
            logger.debug('WARNING: We assume that the old state was "LOCKED_OUT"')
            return LockState.LOCKED_OUT

    async def _set_moria_lock_state(self, state: LockState, telegram_bot) -> None:
        # list with all devices
        devices: List[SurepyDevice] = await self._get_fresh_devices()
        for device in devices:
            # Search for the cat flap
            if device.type == EntityType.CAT_FLAP:
                result_lock = await self.surepy.sac._set_lock_state(device.id, state)
                result_device = await self.surepy.get_device(device.id)
                if result_lock and result_device:
                    telegram_bot.send_text('Done')

    async def unlock_moria(self, msg_sender: MessageSender, _: Any) -> None:
        await self._set_moria_lock_state(LockState.UNLOCKED, msg_sender)

    async def lock_moria_in(self, msg_sender: MessageSender, _: Any) -> None:
        await self._set_moria_lock_state(LockState.LOCKED_IN, msg_sender)

    async def lock_moria_out(self, msg_sender: MessageSender, _: Any) -> None:
        await self._set_moria_lock_state(LockState.LOCKED_OUT, msg_sender)

    async def lock_moria(self, msg_sender: MessageSender, _: Any) -> None:
        await self._set_moria_lock_state(LockState.LOCKED_ALL, msg_sender)

    async def activate_curfew(self, msg_sender: MessageSender, _: Any) -> None:
        await self._set_moria_lock_state(LockState.CURFEW, msg_sender)

    async def lock_moria_curfew(self, msg_sender: MessageSender, _: Any) -> None:
        await self._set_moria_lock_state(LockState.CURFEW_LOCKED, msg_sender)

    async def unlock_moria_curfew(self, msg_sender: MessageSender, _: Any) -> None:
        await self._set_moria_lock_state(LockState.CURFEW_UNLOCKED, msg_sender)

    async def unlock_for_seconds(self, msg_sender: MessageSender, seconds: int) -> None:
        old_state = await self.get_lock_state()
        logger.debug(f"Old state = {old_state}")
        if old_state >= LockState.CURFEW:
            new_state = LockState.CURFEW_UNLOCKED
        else:
            new_state = LockState.LOCKED_IN
        logger.debug(f"New state = {new_state}")
        await self._set_moria_lock_state(new_state, msg_sender)
        await asyncio.sleep(seconds)
        logger.debug(f"Setting back old state = {old_state}")
        await self._set_moria_lock_state(old_state, msg_sender)

    async def switch_pet_location(self, telegram_bot, pet_id: int) -> None:
        pets: List[Dict[str, Any]] = await self.surepy.sac.get_pets()
        if pets is None:
            telegram_bot.send_text(f"No pet was found int he server")
            return

        chosen_pet: Dict[str, Any] | None = None
        for pet in pets:
            if pet["id"] == pet_id:
                chosen_pet = pet
                break

        if chosen_pet is None:
            telegram_bot.send_text(f"Pet with id = '{pet_id}' could not be found")
            return

        old_location: Location = Location(chosen_pet['status']['activity']['where'])
        logger.debug(f"Pet: id= '{chosen_pet['id']}', name= '{chosen_pet['name']}', old location = '{old_location}'")
        if old_location == Location.INSIDE:
            new_location = Location.OUTSIDE
        else:
            new_location = Location.INSIDE
        await self.surepy.sac.set_pet_location(pet_id, new_location)
        telegram_bot.send_text(f"Pet with name = '{chosen_pet['name']}' was marked as '{new_location}'")

    # Helper function used to get fresh data from the devices, so the states are NOT cached by surepy library
    async def _get_fresh_devices(self) -> List[SurepyDevice]:
        return [
            device
            for device in (await self.surepy.get_entities(refresh=True)).values()
            if isinstance(device, SurepyDevice)
        ]
