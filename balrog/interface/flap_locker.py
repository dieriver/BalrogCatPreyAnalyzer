import asyncio
import builtins
import datetime
import os
from typing import Any, Dict, List, Optional

import pytz
from surepy import Surepy, SurepyEntity, SurepyDevice, EntityType
from surepy.entities.devices import Flap
from surepy.entities.pet import Pet
from surepy.enums import LockState, Location

from balrog.config import general_config, flap_config
from balrog.utils.utils import logger


class FlapLocker:
    def __init__(self):
        # user/password authentication (gets a token in background)
        if os.getenv('SUREPET_USER') == "":
            raise Exception("Surepet username not set!. Please set the 'SUREPET_USER' environment variable")
        if os.getenv('SUREPET_PASSWORD') == "":
            raise Exception("Surepet password not set!. Please set the 'SUREPET_PASSWORD' environment variable")
        self.surepy = Surepy(email=os.getenv('SUREPET_USER'), password=os.getenv('SUREPET_PASSWORD'))
        self.old_state: Optional[LockState] = None

    # Functions used to "introspect" the information about pets and devices
    # to register commands
    async def get_pets_data(self) -> Dict[str, int]:
        try:
            registered_pets: List[Pet] = await asyncio.wait_for(
                self.surepy.get_pets(),
                timeout=flap_config.server_timeout_seconds
            )
            pets_data: Dict[str, int] = dict()
            for registered_pet in registered_pets:
                pets_data[registered_pet.name.lower()] = registered_pet.pet_id
        except asyncio.TimeoutError:
            pets_data = {}
        return pets_data

    async def get_devices_data(self) -> Dict[str, int]:
        try:
            # In this case, we are only interested on the name and id of the devices;
            # we can use a cached answer for this
            registered_devices: List[SurepyDevice] = await asyncio.wait_for(
                self.surepy.get_devices(),
                timeout=flap_config.server_timeout_seconds
            )
            devices_data: Dict[str, int] = dict()
            for registered_device in registered_devices:
                devices_data[registered_device.name.lower()] = registered_device.id
        except asyncio.TimeoutError:
            devices_data = {}
        return devices_data

    @staticmethod
    def _parse_pet_data(pet: Pet) -> Optional[str]:
        match type(pet.activity.since):
            case builtins.str:
                location_since = datetime.datetime.fromisoformat(str(pet.activity.since))
            case datetime.datetime:
                location_since = pet.activity.since
            case _:
                location_since = datetime.datetime(1970, 1, 1, 0, 0, 0)
        corrected_since: datetime = location_since.astimezone(pytz.timezone(general_config.local_timezone))
        return (f"\nPet '{pet.name}', Location: {pet.location}, "
                f"Since: {corrected_since.strftime(general_config.timestamp_format)}")

    # Functions used to send data from surepy to the telegram interface
    async def get_pets_status_str(self, filter_by_id: Optional[int] = None) -> str:
        try:
            # list with all pets
            pets: List[Pet] = await asyncio.wait_for(
                self._get_fresh_pets(),
                timeout=flap_config.server_timeout_seconds
            )
            message = f"I found this:"
            for pet in pets:
                if filter_by_id is not None and pet.pet_id != filter_by_id:
                    continue
                else:
                    message += FlapLocker._parse_pet_data(pet)
        except asyncio.TimeoutError:
            message = "Surepet server did not respond in time."
        return message

    async def get_device_data_str(self, device_id: int) -> str:
        # In this case, we need the fresh data of the device
        devices: List[SurepyDevice] = await self._get_fresh_devices()
        for device in devices:
            if device.id == device_id:
                if isinstance(device, Flap):
                    lock_status = device.state
                else:
                    lock_status = LockState.UNLOCKED
                lock_status_str = str(lock_status).replace('_', ' ')
                return f"I found this:\n"\
                       f"Device: '{device.name}'\n"\
                       f"Lock State: '{lock_status_str}'\n"\
                       f"Battery Level: '{device.battery_level}'"
        return "I could not find the device"

    async def list_devices(self) -> str:
        # all entities as id-indexed dict
        entities: Dict[int, SurepyEntity] = await self.surepy.get_entities()

        # list with all devices, with fresh info (?)
        devices: List[SurepyDevice] = await self._get_fresh_devices()
        devices_str = ""
        for device in devices:
            devices_str += f"{device.name = } | {device.serial = } | {device.battery_level = }"
            devices_str += f"{device.type = } | {device.unique_id = } | {device.id = }"
            devices_str += f"{entities[device.parent_id].full_name = } | {entities[device.parent_id] = }\n"
        return devices_str

    async def get_lock_state(self, device_id: int) -> LockState:
        try:
            device: SurepyDevice = await self._get_fresh_device(device_id)
            if device is None:
                # We assume a default value;
                logger.debug('WARNING: No device was found; we assume that the old state was "LOCKED_OUT"')
                return LockState.LOCKED_OUT

            assert device.type == EntityType.CAT_FLAP or device.type == EntityType.PET_FLAP
            assert isinstance(device, Flap)
            return device.state
        except Exception:
            logger.exception('+++ Exception while getting last flap state: ')
            # We assume a default value;
            logger.debug('WARNING: We assume that the old state was "LOCKED_OUT"')
            return LockState.LOCKED_OUT

    async def _set_flap_lock_state(self, device_id: int, state: LockState) -> bool:
        # list with all devices
        try:
            device: SurepyDevice = await asyncio.wait_for(
                self.surepy.get_device(device_id),
                timeout=flap_config.server_timeout_seconds
            )
            if device is None:
                return False

            assert device.type == EntityType.CAT_FLAP or device.type == EntityType.PET_FLAP
            return await asyncio.wait_for(
                self.surepy.sac._set_lock_state(device.id, state),
                timeout=flap_config.server_timeout_seconds
            )
        except asyncio.TimeoutError:
            return False

    async def unlock_device(self, device_id: int) -> bool:
        return await self._set_flap_lock_state(device_id, LockState.UNLOCKED)

    async def device_lock_in(self, device_id: int) -> bool:
        return await self._set_flap_lock_state(device_id, LockState.LOCKED_IN)

    async def device_lock_out(self, device_id: int) -> bool:
        return await self._set_flap_lock_state(device_id, LockState.LOCKED_OUT)

    async def device_lock(self, device_id: int) -> bool:
        return await self._set_flap_lock_state(device_id, LockState.LOCKED_ALL)

    async def device_curfew(self, device_id: int) -> bool:
        return await self._set_flap_lock_state(device_id, LockState.CURFEW)

    async def device_lock_curfew(self, device_id: int) -> bool:
        return await self._set_flap_lock_state(device_id, LockState.CURFEW_LOCKED)

    async def device_unlock_curfew(self, device_id: int) -> bool:
        return await self._set_flap_lock_state(device_id, LockState.CURFEW_UNLOCKED)

    async def unlock_flap_for_let_in(self, device_id: int) -> bool:
        self.old_state = await self.get_lock_state(device_id)
        logger.debug(f"Old state = {self.old_state}")
        if self.old_state >= LockState.CURFEW:
            new_state = LockState.CURFEW_UNLOCKED
        else:
            new_state = LockState.LOCKED_IN
        logger.debug(f"New state = {new_state}")
        return await self._set_flap_lock_state(device_id, new_state)

    async def finish_letin(self, device_id: int) -> bool:
        if self.old_state is not None:
            logger.debug(f"Setting back old state = {self.old_state}")
            return await self._set_flap_lock_state(device_id, self.old_state)
        self.old_state = None

    async def switch_pet_location(self, pet_id: int) -> str:
        try:
            pets: List[Dict[str, Any]] = await asyncio.wait_for(
                self.surepy.sac.get_pets(),
                timeout=flap_config.server_timeout_seconds
            )
        except asyncio.TimeoutError:
            return "Server timed out when retrieving pets."
        if pets is None:
            return "No pet was found in the server"

        chosen_pet: Dict[str, Any] | None = None
        for pet in pets:
            if pet["id"] == pet_id:
                chosen_pet = pet
                break

        if chosen_pet is None:
            return f"Pet with id '{pet_id}' could not be found"

        old_location: Location = Location(chosen_pet['status']['activity']['where'])
        logger.debug(f"Pet: id= '{chosen_pet['id']}', name= '{chosen_pet['name']}', old location = '{old_location}'")
        if old_location == Location.INSIDE:
            new_location = Location.OUTSIDE
        else:
            new_location = Location.INSIDE
        try:
            await asyncio.wait_for(
                self.surepy.sac.set_pet_location(pet_id, new_location),
                timeout=flap_config.server_timeout_seconds
            )
        except asyncio.TimeoutError:
            return f"Server timed out when marking pet '{chosen_pet['name']}' as '{new_location}'"
        return f"Pet '{chosen_pet['name']}' was marked as '{new_location}'"

    # Helper function used to get fresh data from the devices, so the states are NOT cached by surepy library
    async def _get_fresh_devices(self) -> List[SurepyDevice]:
        try:
            entities: Dict[int, SurepyEntity] = await asyncio.wait_for(
                self.surepy.get_entities(refresh=True),
                timeout=flap_config.server_timeout_seconds
            )
        except asyncio.TimeoutError:
            return []
        return [
            device
            for device in entities.values()
            if isinstance(device, SurepyDevice)
        ]

    async def _get_fresh_device(self, device_id: int) -> Optional[SurepyDevice]:
        try:
            entities: Dict[int, SurepyEntity] = await asyncio.wait_for(
                self.surepy.get_entities(refresh=True),
                timeout=flap_config.server_timeout_seconds
            )
        except asyncio.TimeoutError:
            return None
        filter_list = (device for device in entities.values()
                       if isinstance(device, SurepyDevice) and device.id == device_id)
        return next(filter_list, None)

    async def _get_fresh_pets(self) -> List[Pet]:
        try:
            entities: Dict[int, SurepyEntity] = await asyncio.wait_for(
                self.surepy.get_entities(refresh=True),
                timeout=flap_config.server_timeout_seconds
            )
        except asyncio.TimeoutError:
            return []
        return [
            device
            for device in entities.values()
            if isinstance(device, Pet)
        ]
