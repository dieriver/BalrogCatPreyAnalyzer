import asyncio
import os
from surepy import Surepy, SurepyEntity, SurepyDevice, EntityType
from surepy.enums import LockState
from surepy.entities.pet import Pet
from surepy.entities.devices import Flap
from utils import logger
from telegram_bot import NodeBot


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

    async def list_pets(self, telegram_bot: NodeBot):
        # list with all pets
        pets: list[Pet] = await self.surepy.get_pets()
        for pet in pets:
            telegram_bot.send_text(f"\n\n{pet.name}: {pet.state} | {pet.location}\n")

    async def list_devices(self, telegram_bot: NodeBot):
        # all entities as id-indexed dict
        entities: dict[int, SurepyEntity] = await self.surepy.get_entities()

        # list with all devices
        devices: list[SurepyDevice] = await self.surepy.get_devices()
        for device in devices:
            telegram_bot.send_text(f"{device.name = } | {device.serial = } | {device.battery_level = }")
            telegram_bot.send_text(f"{device.type = } | {device.unique_id = } | {device.id = }")
            telegram_bot.send_text(f"{entities[device.parent_id].full_name = } | {entities[device.parent_id] = }\n")

    async def get_lock_state(self):
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

    async def set_moria_lock_state(self, state: LockState, telegram_bot: NodeBot):
        # list with all devices
        devices: list[SurepyDevice] = await self.surepy.get_devices()
        for device in devices:
            # Search for the cat flap
            if device.type == EntityType.CAT_FLAP:
                result_lock = await self.surepy.sac._set_lock_state(device.id, state)
                result_device = await self.surepy.get_device(device.id)
                if result_lock and result_device:
                    telegram_bot.send_text('Done')

    async def unlock_moria(self, telegram_bot: NodeBot):
        await self.set_moria_lock_state(LockState.UNLOCKED, telegram_bot)

    async def lock_moria_in(self, telegram_bot: NodeBot):
        await self.set_moria_lock_state(LockState.LOCKED_IN, telegram_bot)

    async def lock_moria_out(self, telegram_bot: NodeBot):
        await self.set_moria_lock_state(LockState.LOCKED_OUT, telegram_bot)

    async def lock_moria(self, telegram_bot: NodeBot):
        await self.set_moria_lock_state(LockState.LOCKED_ALL, telegram_bot)

    async def activate_curfew(self, telegram_bot: NodeBot):
        await self.set_moria_lock_state(LockState.CURFEW, telegram_bot)

    async def lock_moria_curfew(self, telegram_bot: NodeBot):
        await self.set_moria_lock_state(LockState.CURFEW_LOCKED, telegram_bot)

    async def unlock_moria_curfew(self, telegram_bot: NodeBot):
        await self.set_moria_lock_state(LockState.CURFEW_UNLOCKED, telegram_bot)

    async def unlock_for_seconds(self, telegram_bot: NodeBot, seconds: int):
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
