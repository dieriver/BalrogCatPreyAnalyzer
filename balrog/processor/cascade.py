from dataclasses import dataclass
from logging import DEBUG
from typing import Tuple, Optional

import copy as cpy
import cv2
from cv2.typing import MatLike

from balrog.config import logging_config
from balrog.processor.cv_helpers import draw_rectangle
from balrog.processor.model_stages import PCStage, FFStage, EyeStage, HaarStage, CCMobileNetStage
from balrog.types import Box
from balrog.utils import logger


def _log(level: int, thread_id: int, message: str, exception: Exception | None = None) -> None:
    if exception is not None:
        logger.exception(f"Processor #{thread_id} - {message}")
    elif logging_config.enable_cascade_logging:
        logger.log(level, f"Processor #{thread_id} - {message}")


@dataclass
class EventElement:
    # Inputs
    img_name: str
    raw_image: MatLike
    # Main Output
    output_img: MatLike = None
    # Event preocessing time
    total_time: float = None
    # CC Stage
    pet_detected_sub_img: Box = None
    # Pet
    pet_present: bool = None
    pet_box: Box = None
    # HAAR stage
    haar_face_box: Box = None
    # Face
    face_box: Box = None
    face_bool: bool = None
    # BBS stage
    eyes_box: Box = None
    face_fur_detected: bool = None
    # FF Stage
    face_fur_confidence: float = None
    # PC Stage
    prey_detected: bool = None
    prey_confidence: float = None
    # Inference times
    total_inference_time: float = 0.0
    cc_inference_time: float = None
    haar_inference_time: float = None
    eyes_inference_time: float = None
    ff_bbs_inference_time: float = None
    pc_inference_time: float = None

    def __repr__(self):
        return f"[Evnt-elem: '{self.img_name}', data: '{'ABSENT' if self.raw_image is None else 'Present'}'"


def _do_cc_mobile_stage(
        cc_mobile_stage: CCMobileNetStage,
        cc_target_img: MatLike
) -> Tuple[bool, Optional[Box], Optional[MatLike], float]:
    pet_presence, pet_box, inference_time = cc_mobile_stage.do_cc(target_img=cc_target_img)
    if pet_presence:
        img_xmin = pet_box[0][0]
        img_ymin = pet_box[0][1]
        img_xmax = pet_box[1][0]
        img_ymax = pet_box[1][1]
        detected_pet_img = cc_target_img[img_ymin:img_ymax, img_xmin:img_xmax]
        return pet_presence, pet_box, detected_pet_img, inference_time
    else:
        return pet_presence, None, None, inference_time


def _do_haar_stage(
        haar_stage: HaarStage,
        target_img: MatLike,
        in_box: Box,
        raw_img: MatLike
) -> tuple[bool, MatLike, Box, float]:
    face_found, face_box, haar_inference_time = haar_stage.haar_do(
        sub_img=target_img,
        full_img=raw_img,
        prev_box=in_box
    )
    pc_xmin = int(face_box[0][0])
    pc_ymin = int(face_box[0][1])
    pc_xmax = int(face_box[1][0])
    pc_ymax = int(face_box[1][1])
    face_sub_img = cpy.deepcopy(raw_img[pc_ymin:pc_ymax, pc_xmin:pc_xmax])

    return face_found, face_sub_img, face_box, haar_inference_time


def _do_face_fur_stage(ff_stage: FFStage, in_img: MatLike) -> tuple[bool, float, float]:
    face_fur_detected, face_fur_confidence, ff_inference_time = ff_stage.face_fur_do(target_img=in_img)
    return face_fur_detected, face_fur_confidence, ff_inference_time


def _do_pc_stage(pc_stage: PCStage, pc_target_img: MatLike) -> tuple[bool, float, float]:
    prey_detected, prey_confidence, inference_time = pc_stage.pc_do(target_img=pc_target_img)
    return prey_detected, prey_confidence, inference_time


def _write_text_on_img(img: MatLike, text: str, text_pos: Tuple[int, int],
                       color: Tuple[float, float, float]) -> MatLike:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    line_type = 3

    cv2.putText(img, text, text_pos, font, font_scale, color, line_type)
    return img


class Cascade:
    def __init__(self):
        # Models
        self.cc_mobile_stage = CCMobileNetStage()
        self.pc_stage = PCStage()
        self.ff_stage = FFStage()
        self.eyes_stage = EyeStage()
        self.haar_stage = HaarStage()

    def do_single_cascade(self, event_img_object: EventElement, thread_id: int, frame_index: int) -> None:
        logger.info(f"Processor #{thread_id} - Processing index: '{frame_index}', "
                    f"img_data: {'ABSENT' if event_img_object.raw_image is None else 'Present' }, "
                    f"name: '{event_img_object.img_name}'")
        if event_img_object.raw_image is None:
            return

        original_copy_img = cpy.deepcopy(event_img_object.raw_image)
        copy_img_for_cascade = cpy.deepcopy(event_img_object.raw_image)

        # Do CC - Recognizes pet (either cat or dog)
        pet_present, pet_box, pet_detected_sub_img, cc_inference_time = _do_cc_mobile_stage(
            cc_mobile_stage=self.cc_mobile_stage, cc_target_img=copy_img_for_cascade
        )
        _log(DEBUG, thread_id, f"CASCADE - CC compute Time: {cc_inference_time}")
        event_img_object.pet_present = pet_present
        event_img_object.pet_box = pet_box
        event_img_object.pet_detected_sub_img = pet_detected_sub_img
        event_img_object.cc_inference_time = cc_inference_time
        event_img_object.total_inference_time += cc_inference_time

        if pet_present and pet_detected_sub_img.size != 0:
            _log(DEBUG, thread_id, "CASCADE - Cat Detected!")
            rec_img = draw_rectangle(
                img=original_copy_img,
                box=pet_box,
                color=(255, 0, 0),
                text='CC_Pred'
            )

            # Do HAAR - Recognizes face inside the detected pet box
            face_found, face_sub_img, face_box, haar_inference_time = _do_haar_stage(
                haar_stage=self.haar_stage,
                target_img=pet_detected_sub_img,
                in_box=pet_box,
                raw_img=event_img_object.raw_image
            )
            rec_img = draw_rectangle(
                img=rec_img,
                box=face_box,
                color=(0, 255, 255),
                text='HAAR_Pred'
            )

            event_img_object.haar_face_box = face_box
            event_img_object.haar_inference_time = haar_inference_time
            event_img_object.total_inference_time += haar_inference_time

            overlap = Cascade._cc_haar_overlap(cc_box=pet_box, haar_box=face_box, thread_id=thread_id)
            if face_found and face_sub_img.size != 0 and overlap >= 0.1:
                # Pet and face boxes overlap quite a bit: easy case
                detected_face_box = face_box
                detected_face = True
                cropped_img = face_sub_img
            else:
                # Pet and face boxes do not overlap enough; analyze deeper
                # Do EYES - Recognizes eyes inside the pet box
                eyes_image, eyes_box, eye_inference_time = self.eyes_stage.do_eyes(
                    in_image=pet_detected_sub_img,
                    raw_image=event_img_object.raw_image,
                    in_box=pet_box
                )
                rec_img = draw_rectangle(img=rec_img, box=eyes_box, color=(255, 0, 255), text='BBS_Pred')
                event_img_object.eyes_box = eyes_box
                event_img_object.eyes_inference_time = eye_inference_time
                event_img_object.total_inference_time += eye_inference_time

                # Do FF for Haar and EYES - Recognizes fur and eyes inside the eyes image
                face_fur_detected, face_fur_confidence, ff_inference_time = _do_face_fur_stage(
                    ff_stage=self.ff_stage,
                    in_img=eyes_image
                )
                event_img_object.face_fur_detected = face_fur_detected
                event_img_object.face_fur_confidence = face_fur_confidence
                event_img_object.ff_bbs_inference_time = ff_inference_time
                event_img_object.total_inference_time += ff_inference_time

                detected_face_box = eyes_box
                detected_face = face_fur_detected
                cropped_img = eyes_image

            event_img_object.face_bool = detected_face
            event_img_object.face_box = detected_face_box

            if detected_face:
                rec_img = draw_rectangle(img=rec_img, box=detected_face_box, color=(255, 255, 255), text='INF_Pred')
                _log(DEBUG, thread_id, "CASCADE - Face Detected!")

                # Do PC - Check if there is a prey in the crop image under analysis
                pred_class, pred_val, pc_inference_time = _do_pc_stage(
                    pc_stage=self.pc_stage,
                    pc_target_img=cropped_img
                )
                _log(DEBUG, thread_id, f"CASCADE - Prey Prediction: {pred_class}")
                _log(DEBUG, thread_id, f"CASCADE - Pred_Val: {pred_val:.2f}")
                pc_str = f' Prey: {pred_class} @ {pred_val:.2f}'
                color = (0, 0, 255) if pred_class else (0, 255, 0)
                rec_img = _write_text_on_img(img=rec_img, text=pc_str, text_pos=(15, 100), color=color)

                event_img_object.prey_detected = pred_class
                event_img_object.prey_confidence = pred_val
                event_img_object.pc_inference_time = pc_inference_time
                event_img_object.total_inference_time += pc_inference_time

            else:
                _log(DEBUG, thread_id, "CASCADE - No Face Found...")
                ff_str = 'No_Face'
                rec_img = _write_text_on_img(img=rec_img, text=ff_str, text_pos=(15, 100), color=(255, 255, 0))

        else:
            _log(DEBUG, thread_id, "CASCADE - No Cat Found...")
            rec_img = _write_text_on_img(
                img=original_copy_img,
                text='CC_Pred: NoCat',
                text_pos=(15, 100),
                color=(255, 255, 0)
            )

        # Always save rec_img in event_img object
        event_img_object.output_img = rec_img

    @staticmethod
    def _cc_haar_overlap(cc_box: Box, haar_box: Box, thread_id: int) -> float:
        cc_area = abs(cc_box[0][0] - cc_box[1][0]) * abs(cc_box[0][1] - cc_box[1][1])
        haar_area = abs(haar_box[0][0] - haar_box[1][0]) * abs(haar_box[0][1] - haar_box[1][1])
        overlap = haar_area / cc_area
        _log(DEBUG, thread_id, f"CASCADE - Overlap: {overlap}")
        return overlap
