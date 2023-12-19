import asyncio
import os
from datetime import datetime
from typing import Any

from surepy import Surepy, SurepyEntity, SurepyDevice, EntityType
from surepy.entities.devices import Flap
from surepy.entities.pet import Pet
from surepy.enums import LockState, Location

from balrog.utils.utils import logger


class FlapLocker:
    def __init__(self):
        # user/password authentication (gets a token in background)
        if os.getenv('SUREPET_USER') == "":
            raise Exception("Surepet username not set!. Please set the 'SUREPET_USER' environment variable")
        if os.getenv('SUREPET_PASSWORD') == "":
            raise Exception("Surepet password not set!. Please set the 'SUREPET_PASSWORD' environment variable")
        self.surepy = Surepy(email=os.getenv('SUREPET_USER'), password=os.getenv('SUREPET_PASSWORD'))

        # token authentication (token supplied via SUREPY_TOKEN env var)
        #token = 'XXXXX' # Complete if necessary
        #surepy = Surepy(auth_token=token)

    async def get_pets_data(self) -> dict[str, int]:
        registered_pets: list[Pet] = await self.surepy.get_pets()
        pets_data: dict[str, int] = dict()
        for registered_pet in registered_pets:
            pets_data[registered_pet.name] = registered_pet.pet_id
        return pets_data

    async def list_pets_data(self, telegram_bot) -> None:
        # list with all pets
        pets: list[dict[str, Any]] = await self.surepy.sac.get_pets()
        message = f"I found this:"
        for pet in pets:
            location: Location = Location(pet['status']['activity']['where'])
            location_since: datetime = datetime.fromisoformat(pet['status']['activity']['since'])
            message += (f"\nPet '{pet['name']}', location: {location}, "
                        f"since: {location_since}")
        telegram_bot.send_text(message)

    async def list_devices(self, telegram_bot) -> None:
        # all entities as id-indexed dict
        entities: dict[int, SurepyEntity] = await self.surepy.get_entities()

        # list with all devices
        devices: list[SurepyDevice] = await self.surepy.get_devices()
        for device in devices:
            telegram_bot.send_text(f"{device.name = } | {device.serial = } | {device.battery_level = }")
            telegram_bot.send_text(f"{device.type = } | {device.unique_id = } | {device.id = }")
            telegram_bot.send_text(f"{entities[device.parent_id].full_name = } | {entities[device.parent_id] = }\n")

    async def get_lock_state(self) -> LockState:
        try:
            devices: list[SurepyDevice] = await self.surepy.get_devices()
            for device in devices:
                if device.type == EntityType.CAT_FLAP:
                    cat_flap: Flap = device
                    return cat_flap.state
        except Exception:
            logger.exception('+++ Exception while getting last flap state: ')
            # We assume a default value;
            logger.debug('WARNING: We assume that the old state was "LOCKED_OUT"')
            return LockState.LOCKED_OUT

    async def set_moria_lock_state(self, state: LockState, telegram_bot) -> None:
        # list with all devices
        devices: list[SurepyDevice] = await self.surepy.get_devices()
        for device in devices:
            # Search for the cat flap
            if device.type == EntityType.CAT_FLAP:
                result_lock = await self.surepy.sac._set_lock_state(device.id, state)
                result_device = await self.surepy.get_device(device.id)
                if result_lock and result_device:
                    telegram_bot.send_text('Done')

    async def unlock_moria(self, telegram_bot) -> None:
        await self.set_moria_lock_state(LockState.UNLOCKED, telegram_bot)

    async def lock_moria_in(self, telegram_bot) -> None:
        await self.set_moria_lock_state(LockState.LOCKED_IN, telegram_bot)

    async def lock_moria_out(self, telegram_bot) -> None:
        await self.set_moria_lock_state(LockState.LOCKED_OUT, telegram_bot)

    async def lock_moria(self, telegram_bot):
        await self.set_moria_lock_state(LockState.LOCKED_ALL, telegram_bot)

    async def activate_curfew(self, telegram_bot) -> None:
        await self.set_moria_lock_state(LockState.CURFEW, telegram_bot)

    async def lock_moria_curfew(self, telegram_bot) -> None:
        await self.set_moria_lock_state(LockState.CURFEW_LOCKED, telegram_bot)

    async def unlock_moria_curfew(self, telegram_bot) -> None:
        await self.set_moria_lock_state(LockState.CURFEW_UNLOCKED, telegram_bot)

    async def unlock_for_seconds(self, telegram_bot, seconds: int) -> None:
        old_state = await self.get_lock_state()
        logger.debug(f"Old state = {old_state}")
        if old_state >= LockState.CURFEW:
            new_state = LockState.CURFEW_UNLOCKED
        else:
            new_state = LockState.UNLOCKED
        logger.debug(f"New state = {new_state}")
        await self.set_moria_lock_state(new_state, telegram_bot)
        await asyncio.sleep(seconds)
        logger.debug(f"Setting back old state = {old_state}")
        await self.set_moria_lock_state(old_state, telegram_bot)

    async def switch_pet_location(self, telegram_bot, pet_id: int) -> None:
        pets: list[dict[str, Any]] = await self.surepy.sac.get_pets()
        if pets is None:
            telegram_bot.send_text(f"No pet was found int he server")
            return

        chosen_pet: dict[str, Any] | None = None
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
