import logging
from dataclasses import dataclass
from tomllib import load


@dataclass
class GeneralConfigs:
    max_message_sender_threads: int
    max_frame_processor_threads: int
    min_aggregation_frames_threshold: int
    max_frame_buffers: int


@dataclass
class LoggingConfigs:
    log_base_folder: str
    log_file_name: str
    log_dbg_file_name: str
    stdout_debug_level: int
    enable_cascade_logging: bool
    enable_circular_buffer_logging: bool
    max_log_file_size_mb: int
    max_log_files_kept: int


@dataclass
class CameraConfigs:
    camera_fps: int
    camera_cleanup_frames_threshold: int


@dataclass
class ModelConfigs:
    tensorflow_models_path: str
    event_reset_threshold: int
    cat_counter_threshold: int
    cumulus_prey_threshold: int
    cumulus_no_prey_threshold: int
    prey_val_hard_threshold: int


@dataclass
class FlapConfigs:
    let_in_open_seconds: int


def load_flap_config() -> FlapConfigs:
    with open("../config.toml", "rb") as config_file:
        loaded_bytes = load(config_file)
        return FlapConfigs(
            loaded_bytes["flap"]["let_in_open_seconds"],
        )


def load_general_config() -> GeneralConfigs:
    with open("../config.toml", "rb") as config_file:
        loaded_bytes = load(config_file)
        return GeneralConfigs(
            loaded_bytes["general"]["max_message_sender_threads"],
            loaded_bytes["general"]["max_frame_processor_threads"],
            loaded_bytes["general"]["min_aggregation_frames_threshold"],
            loaded_bytes["general"]["max_frame_buffers"],
        )


def load_logging_config() -> LoggingConfigs:
    with open("../config.toml", "rb") as config_file:
        loaded_bytes = load(config_file)
        return LoggingConfigs(
            loaded_bytes["logging"]["log_base_folder"],
            loaded_bytes["logging"]["log_file_name"],
            loaded_bytes["logging"]["log_dbg_file_name"],
            logging.getLevelName(loaded_bytes["logging"]["stdout_debug_level"]),
            loaded_bytes["logging"]["enable_cascade_logging"],
            loaded_bytes["logging"]["enable_circular_buffer_logging"],
            loaded_bytes["logging"]["max_log_file_size_mb"],
            loaded_bytes["logging"]["max_log_files_kept"]
        )


def load_camera_config() -> CameraConfigs:
    with open("../config.toml", "rb") as config_file:
        loaded_bytes = load(config_file)
        return CameraConfigs(
            loaded_bytes["camera"]["camera_fps"],
            loaded_bytes["camera"]["camera_cleanup_frames_threshold"]
        )


def load_model_config() -> ModelConfigs:
    with open("../config.toml", "rb") as config_file:
        loaded_bytes = load(config_file)
        return ModelConfigs(
            loaded_bytes["model"]["tensorflow_models_path"],
            loaded_bytes["model"]["event_reset_threshold"],
            loaded_bytes["model"]["cat_counter_threshold"],
            loaded_bytes["model"]["cumulus_prey_threshold"],
            loaded_bytes["model"]["cumulus_no_prey_threshold"],
            loaded_bytes["model"]["prey_val_hard_threshold"]
        )


general_config = load_general_config()
logging_config = load_logging_config()
model_config = load_model_config()
camera_config = load_camera_config()
flap_config = load_flap_config()
