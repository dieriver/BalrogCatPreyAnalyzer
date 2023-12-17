from dataclasses import dataclass

from cv2.typing import MatLike

from .cascade import Cascade
from .image_container import ImageContainer, ImageBuffers


@dataclass
class EventElement:
        img_name: str
        cc_target_img: MatLike
        cc_cat_bool = None
        cc_pred_bb = None
        cc_inference_time = None
        cr_class = None
        cr_val = None
        cr_inference_time = None
        bbs_target_img = None
        bbs_pred_bb = None
        bbs_inference_time = None
        haar_pred_bb = None
        haar_inference_time = None
        ff_haar_bool = None
        ff_haar_val = None
        ff_haar_inference_time = None
        ff_bbs_bool = None
        ff_bbs_val = None
        ff_bbs_inference_time = None
        face_box = None
        face_bool = None
        pc_prey_class = None
        pc_prey_val = None
        pc_inference_time = None
        total_inference_time = None
        output_img = None