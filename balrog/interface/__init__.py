import collections
from abc import ABC, abstractmethod
from multiprocessing import Event
from threading import Thread
from typing import Self, Optional, Deque

from cv2.typing import MatLike


class MessageSender(ABC):
    def __init__(self):
        # Data coming and used form unexpected places (other files)
        self._node_live_img: Optional[MatLike] = None
        self._node_last_casc_img: Optional[MatLike] = None
        self._frames_rdy_for_img: Optional[int] = None
        self._frames_rdy_for_casc: Optional[int] = None
        self._frames_rdy_for_agg: Optional[int] = None
        self._node_over_head_info: Optional[float] = None
        self._last_delays: Deque[float] = collections.deque(maxlen=20)
        self._mute_images: bool = False
        self.sender_thread: Optional[Thread] = None

    @classmethod
    def get_message_sender_instance(
            cls,
            is_debug: bool = False,
            stop_event: Event = None
    ) -> Self:
        if is_debug:
            from balrog.interface.telegram_bot import DebugBot
            return DebugBot()
        else:
            from balrog.interface.telegram_bot import BalrogTelegramBot
            return BalrogTelegramBot(stop_event)

    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass

    @abstractmethod
    def send_text(self, message: str) -> None:
        pass

    @abstractmethod
    def send_img(self, img: MatLike, caption: str) -> None:
        pass

    @property
    def live_img(self) -> MatLike | None:
        return self._node_live_img

    @live_img.setter
    def live_img(self, node_live_img: MatLike) -> None:
        self._node_live_img = node_live_img.copy()

    @property
    def last_casc_img(self) -> MatLike | None:
        return self._node_last_casc_img

    @last_casc_img.setter
    def last_casc_img(self, node_last_casc_img: MatLike) -> None:
        self._node_last_casc_img = node_last_casc_img.copy()

    @property
    def frames_rdy_for_img(self) -> int:
        return self._frames_rdy_for_img

    @frames_rdy_for_img.setter
    def frames_rdy_for_img(self, frames_rdy_for_img: int) -> None:
        self._frames_rdy_for_img = frames_rdy_for_img

    @property
    def frames_rdy_for_cascade(self) -> int:
        return self._frames_rdy_for_casc

    @frames_rdy_for_cascade.setter
    def frames_rdy_for_cascade(self, frames_rdy_for_cascade: int) -> None:
        self._frames_rdy_for_casc = frames_rdy_for_cascade

    @property
    def frames_rdy_for_aggregate(self) -> int | None:
        return self._frames_rdy_for_agg

    @frames_rdy_for_aggregate.setter
    def frames_rdy_for_aggregate(self, frames_rdy_for_agg: int) -> None:
        self._frames_rdy_for_agg = frames_rdy_for_agg

    @property
    def last_casc_time(self):
        return self._node_over_head_info

    @last_casc_time.setter
    def last_casc_time(self, node_over_head_info: float) -> None:
        self._node_over_head_info = node_over_head_info

    @property
    def queue_avg_delay(self):
        if len(self._last_delays) == 0:
            return None
        return float(sum(self._last_delays)) / float(len(self._last_delays))

    def add_delay(self, val: float) -> None:
        self._last_delays.append(val)

    @property
    def muted_images(self) -> bool:
        return self._mute_images

    @muted_images.setter
    def muted_images(self, value: bool) -> None:
        self._mute_images = value
