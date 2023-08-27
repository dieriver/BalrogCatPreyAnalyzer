import asyncio
import logging
import os
from typing import List, Dict
from surepy import Surepy, SurepyEntity, SurepyDevice, EntityType
from surepy.enums import LockState
from surepy.entities.pet import Pet
from surepy.entities.devices import Flap

logger = logging.getLogger("cat_logger")


class FlapLocker:
    def __init__(self):
        # user/password authentication (gets a token in background)
        if os.getenv('SUREPET_USER') == "":
            raise Exception("Surepet username not set!. Please set the 'SUREPET_USER' environment variable")
        if os.getenv('SUREPET_PASSWORD') == "":
            raise Exception("Surepet password not set!. Please set the 'SUREPET_PASSWORD' environment variable")
        self.surepy = Surepy(email=os.getenv('SUREPET_USER'), password=os.getenv('SUREPET_PASSWORD'))
        self.old_state = LockState.UNLOCKED

        # token authentication (token supplied via SUREPY_TOKEN env var)
        #token = 'XXXXX' # Complete if necessary
        #surepy = Surepy(auth_token=token)

    async def list_pets(self, telegram_bot):
        # list with all pets
        pets: List[Pet] = await self.surepy.get_pets()
        for pet in pets:
            telegram_bot.send_text(f"\n\n{pet.name}: {pet.state} | {pet.location}\n")

    async def list_devices(self, telegram_bot):
        # all entities as id-indexed dict
        entities: Dict[int, SurepyEntity] = await self.surepy.get_entities()

        # list with all devices
        devices: List[SurepyDevice] = await self.surepy.get_devices()
        for device in devices:
            telegram_bot.send_text(f"{device.name = } | {device.serial = } | {device.battery_level = }")
            telegram_bot.send_text(f"{device.type = } | {device.unique_id = } | {device.id = }")
            telegram_bot.send_text(f"{entities[device.parent_id].full_name = } | {entities[device.parent_id] = }\n")

    async def get_lock_state(self):
        devices: List[SurepyDevice] = await self.surepy.get_devices()
        for device in devices:
            if device.type == EntityType.CAT_FLAP:
                cat_flap: Flap = device;
                return cat_flap.state

    async def set_moria_lock_state(self, state: LockState, telegram_bot):
        # list with all devices
        devices: List[SurepyDevice] = await self.surepy.get_devices()
        for device in devices:
            # Search for the cat flap
            if device.type == EntityType.CAT_FLAP:
                result_lock = await self.surepy.sac._set_lock_state(device.id, state)
                result_device = await self.surepy.get_device(device.id)
                if result_lock and result_device:
                    telegram_bot.send_text('Done')

    async def unlock_moria(self, telegram_bot):
        await self.set_moria_lock_state(LockState.UNLOCKED, telegram_bot)

    async def lock_moria_in(self, telegram_bot):
        await self.set_moria_lock_state(LockState.LOCKED_IN, telegram_bot)

    async def lock_moria_out(self, telegram_bot):
        await self.set_moria_lock_state(LockState.LOCKED_OUT, telegram_bot)

    async def lock_moria(self, telegram_bot):
        await self.set_moria_lock_state(LockState.LOCKED_ALL, telegram_bot)

    async def activate_curfew(self, telegram_bot):
        await self.set_moria_lock_state(LockState.CURFEW, telegram_bot)

    async def lock_moria_curfew(self, telegram_bot):
        await self.set_moria_lock_state(LockState.CURFEW_LOCKED, telegram_bot)

    async def unlock_moria_curfew(self, telegram_bot):
        await self.set_moria_lock_state(LockState.CURFEW_UNLOCKED, telegram_bot)

    async def unlock_for_seconds(self, telegram_bot, seconds: int):
        self.old_state = await self.get_lock_state()
        if self.old_state >= LockState.CURFEW:
            new_state = LockState.CURFEW_UNLOCKED
        else:
            new_state = LockState.UNLOCKED
        await self.set_moria_lock_state(new_state, telegram_bot)
        await asyncio.sleep(seconds)
        await self.set_moria_lock_state(self.old_state, telegram_bot)
        self.old_state = LockState.UNLOCKED