from dataclasses import dataclass
from tomllib import load


@dataclass
class GeneralConfigs:
    max_message_sender_threads: int
    max_frame_processor_threads: int
    min_aggregation_frames_threshold: int
    max_frame_buffers: int
    camera_fps: int
    camera_cleanup_frames_threshold: int


@dataclass
class ModelConfigs:
    event_reset_threshold: int
    cat_counter_threshold: int
    cumulus_prey_threshold: int
    cumulus_no_prey_threshold: int
    prey_val_hard_threshold: int


def load_model_config() -> ModelConfigs:
    with open("config.toml", "rb") as config_file:
        loaded_bytes = load(config_file)
        return ModelConfigs(
            loaded_bytes["general"]["event_reset_threshold"],
            loaded_bytes["general"]["cat_counter_threshold"],
            loaded_bytes["general"]["cumulus_prey_threshold"],
            loaded_bytes["general"]["cumulus_no_prey_threshold"],
            loaded_bytes["general"]["prey_val_hard_threshold"]
        )


def load_general_config() -> GeneralConfigs:
    with open("config.toml", "rb") as config_file:
        loaded_bytes = load(config_file)
        return GeneralConfigs(
            loaded_bytes["general"]["max_message_sender_threads"],
            loaded_bytes["general"]["max_frame_processor_threads"],
            loaded_bytes["general"]["min_aggregation_frames_threshold"],
            loaded_bytes["general"]["max_frame_buffers"],
            loaded_bytes["general"]["camera_fps"],
            loaded_bytes["general"]["camera_cleanup_frames_threshold"]
        )


general_config = load_general_config()
model_config = load_model_config()
