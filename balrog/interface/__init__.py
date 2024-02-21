from abc import ABC, abstractmethod
from multiprocessing import Event
from typing import Self, Optional

from cv2.typing import MatLike


class MessageSender(ABC):
    def __init__(self):
        # Data coming and used form unexpected places (other files)
        self._node_live_img: Optional[MatLike] = None
        self._node_last_casc_img: Optional[MatLike] = None
        self._node_queue_info: Optional[int] = None
        self._node_over_head_info: Optional[float] = None
        self._mute_images: bool = False

    @classmethod
    def get_message_sender_instance(
            cls,
            is_debug: bool = False,
            clean_queue_event: Event = None,
            stop_event: Event = None
    ) -> Self:
        if is_debug:
            from balrog.interface.telegram_bot import DebugBot
            return DebugBot()
        else:
            from balrog.interface.telegram_bot import BalrogTelegramBot
            return BalrogTelegramBot(clean_queue_event, stop_event)

    @abstractmethod
    def send_text(self, message: str) -> None:
        pass

    @abstractmethod
    def send_img(self, img: MatLike, caption: str) -> None:
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

    @property
    def muted_images(self) -> bool:
        return self._mute_images

    @muted_images.setter
    def muted_images(self, value: bool) -> None:
        self._mute_images = value
