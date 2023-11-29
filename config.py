from dataclasses import dataclass
from tomllib import load


@dataclass
class GeneralConfigs:
    max_message_sender_threads: int
    max_frame_processor_threads: int
    min_aggregation_frames_threshold: int
    max_frame_buffers: int


@dataclass
class CameraConfigs:
    camera_fps: int
    camera_cleanup_frames_threshold: int


@dataclass
class ModelConfigs:
    event_reset_threshold: int
    cat_counter_threshold: int
    cumulus_prey_threshold: int
    cumulus_no_prey_threshold: int
    prey_val_hard_threshold: int


@dataclass
class FlapConfigs:
    let_in_open_seconds: int


def load_flap_config() -> FlapConfigs:
    with open("config.toml", "rb") as config_file:
        loaded_bytes = load(config_file)
        return FlapConfigs(
            loaded_bytes["flap"]["let_in_open_seconds"],
        )


def load_general_config() -> GeneralConfigs:
    with open("config.toml", "rb") as config_file:
        loaded_bytes = load(config_file)
        return GeneralConfigs(
            loaded_bytes["general"]["max_message_sender_threads"],
            loaded_bytes["general"]["max_frame_processor_threads"],
            loaded_bytes["general"]["min_aggregation_frames_threshold"],
            loaded_bytes["general"]["max_frame_buffers"],
        )


def load_camera_config() -> CameraConfigs:
    with open("config.toml", "rb") as config_file:
        loaded_bytes = load(config_file)
        return CameraConfigs(
            loaded_bytes["camera"]["camera_fps"],
            loaded_bytes["camera"]["camera_cleanup_frames_threshold"]
        )


def load_model_config() -> ModelConfigs:
    with open("config.toml", "rb") as config_file:
        loaded_bytes = load(config_file)
        return ModelConfigs(
            loaded_bytes["model"]["event_reset_threshold"],
            loaded_bytes["model"]["cat_counter_threshold"],
            loaded_bytes["model"]["cumulus_prey_threshold"],
            loaded_bytes["model"]["cumulus_no_prey_threshold"],
            loaded_bytes["model"]["prey_val_hard_threshold"]
        )


general_config = load_general_config()
model_config = load_model_config()
camera_config = load_camera_config()
flap_config = load_flap_config()
