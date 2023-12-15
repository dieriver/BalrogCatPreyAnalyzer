import time

import cv2
import numpy as np

from balrog.config import logging_config
from balrog.utils.utils import logger
from .model_stages import PCStage, FFStage, EyeStage, HaarStage, CCMobileNetStage


class EventElement:
    def __init__(self, img_name, cc_target_img):
        self.img_name = img_name
        self.cc_target_img = cc_target_img
        self.cc_cat_bool = None
        self.cc_pred_bb = None
        self.cc_inference_time = None
        self.cr_class = None
        self.cr_val = None
        self.cr_inference_time = None
        self.bbs_target_img = None
        self.bbs_pred_bb = None
        self.bbs_inference_time = None
        self.haar_pred_bb = None
        self.haar_inference_time = None
        self.ff_haar_bool = None
        self.ff_haar_val = None
        self.ff_haar_inference_time = None
        self.ff_bbs_bool = None
        self.ff_bbs_val = None
        self.ff_bbs_inference_time = None
        self.face_box = None
        self.face_bool = None
        self.pc_prey_class = None
        self.pc_prey_val = None
        self.pc_inference_time = None
        self.total_inference_time = None
        self.output_img = None


class Cascade:
    def __init__(self):
        # Models
        self.cc_mobile_stage = CCMobileNetStage()
        self.pc_stage = PCStage()
        self.ff_stage = FFStage()
        self.eyes_stage = EyeStage()
        self.haar_stage = HaarStage()

    @staticmethod
    def _log(message: str, exception: Exception | None = None):
        if exception is not None:
            logger.exception(message)
        elif logging_config.enable_cascade_logging:
            logger.debug(message)

    def do_single_cascade(self, event_img_object: EventElement, thread_id: int):
        logger.info(f'Thread {thread_id} - Processing image: {event_img_object.img_name}')
        cc_target_img = event_img_object.cc_target_img
        original_copy_img = cc_target_img.copy()

        # Do CC
        start_time = time.time()
        dk_bool, cat_bool, bbs_target_img, pred_cc_bb_full, cc_inference_time = self.do_cc_mobile_stage(
            cc_target_img=cc_target_img)
        current_time = time.time()
        Cascade._log(f'CASCADE - CC compute Time: {current_time - start_time}')
        event_img_object.cc_cat_bool = cat_bool
        event_img_object.cc_pred_bb = pred_cc_bb_full
        event_img_object.bbs_target_img = bbs_target_img
        event_img_object.cc_inference_time = cc_inference_time

        if cat_bool and bbs_target_img.size != 0:
            Cascade._log('CASCADE - Cat Detected!')
            rec_img = self.cc_mobile_stage.draw_rectangle(img=original_copy_img,
                                                          box=pred_cc_bb_full,
                                                          color=(255, 0, 0),
                                                          text='CC_Pred')

            # Do HAAR
            haar_snout_crop, haar_bbs, haar_inference_time, haar_found_bool = (
                self.do_haar_stage(
                    target_img=bbs_target_img,
                    pred_cc_bb_full=pred_cc_bb_full,
                    cc_target_img=cc_target_img
                )
            )
            rec_img = self.cc_mobile_stage.draw_rectangle(
                img=rec_img,
                box=haar_bbs,
                color=(0, 255, 255),
                text='HAAR_Pred'
            )

            event_img_object.haar_pred_bb = haar_bbs
            event_img_object.haar_inference_time = haar_inference_time

            if haar_found_bool and haar_snout_crop.size != 0 and self.cc_haar_overlap(cc_bbs=pred_cc_bb_full,
                                                                                      haar_bbs=haar_bbs) >= 0.1:
                inf_bb = haar_bbs
                face_bool = True
                snout_crop = haar_snout_crop

            else:
                # Do EYES
                bbs_snout_crop, bbs, eye_inference_time = self.do_eyes_stage(eye_target_img=bbs_target_img,
                                                                             cc_pred_bb=pred_cc_bb_full,
                                                                             cc_target_img=cc_target_img)
                rec_img = self.cc_mobile_stage.draw_rectangle(img=rec_img, box=bbs, color=(255, 0, 255),
                                                              text='BBS_Pred')
                event_img_object.bbs_pred_bb = bbs
                event_img_object.bbs_inference_time = eye_inference_time

                # Do FF for Haar and EYES
                bbs_dk_bool, bbs_face_bool, bbs_ff_conf, bbs_ff_inference_time = self.do_ff_stage(
                    snout_crop=bbs_snout_crop)
                event_img_object.ff_bbs_bool = bbs_face_bool
                event_img_object.ff_bbs_val = bbs_ff_conf
                event_img_object.ff_bbs_inference_time = bbs_ff_inference_time

                inf_bb = bbs
                face_bool = bbs_face_bool
                snout_crop = bbs_snout_crop

            event_img_object.face_bool = face_bool
            event_img_object.face_box = inf_bb

            if face_bool:
                rec_img = self.cc_mobile_stage.draw_rectangle(img=rec_img, box=inf_bb, color=(255, 255, 255),
                                                              text='INF_Pred')
                Cascade._log('CASCADE - Face Detected!')

                # Do PC
                pred_class, pred_val, inference_time = self.do_pc_stage(pc_target_img=snout_crop)
                Cascade._log(f'CASCADE - Prey Prediction: {pred_class}')
                Cascade._log(f'CASCADE - Pred_Val: {pred_val:.2f}')
                pc_str = f' PC_Pred: {pred_class} @ {pred_val:.2f}'
                color = (0, 0, 255) if pred_class else (0, 255, 0)
                rec_img = self.input_text(img=rec_img, text=pc_str, text_pos=(15, 100), color=color)

                event_img_object.pc_prey_class = pred_class
                event_img_object.pc_prey_val = pred_val
                event_img_object.pc_inference_time = inference_time

            else:
                Cascade._log('CASCADE - No Face Found...')
                ff_str = 'No_Face'
                rec_img = self.input_text(img=rec_img, text=ff_str, text_pos=(15, 100), color=(255, 255, 0))

        else:
            Cascade._log('CASCADE - No Cat Found...')
            rec_img = self.input_text(img=original_copy_img,
                                      text='CC_Pred: NoCat',
                                      text_pos=(15, 100),
                                      color=(255, 255, 0)
                                      )

        # Always save rec_img in event_img object
        event_img_object.output_img = rec_img
        return event_img_object

    def cc_haar_overlap(self, cc_bbs, haar_bbs):
        cc_area = abs(cc_bbs[0][0] - cc_bbs[1][0]) * abs(cc_bbs[0][1] - cc_bbs[1][1])
        haar_area = abs(haar_bbs[0][0] - haar_bbs[1][0]) * abs(haar_bbs[0][1] - haar_bbs[1][1])
        overlap = haar_area / cc_area
        Cascade._log(f'CASCADE - Overlap: {overlap}')
        return overlap

    def infere_snout_crop(self, bbs, haar_bbs, bbs_face_bool, bbs_ff_conf, haar_face_bool, haar_ff_conf, cc_target_img):
        # Combine BBS's if both are faces
        if bbs_face_bool and haar_face_bool:
            xmin = min(bbs[0][0], haar_bbs[0][0])
            ymin = min(bbs[0][1], haar_bbs[0][1])
            xmax = max(bbs[1][0], haar_bbs[1][0])
            ymax = max(bbs[1][1], haar_bbs[1][1])
            inf_bb = np.array([(xmin, ymin), (xmax, ymax)]).reshape((-1, 2))
            snout_crop = cc_target_img[ymin:ymax, xmin:xmax]
            return snout_crop, inf_bb, False, True, (bbs_ff_conf + haar_ff_conf) / 2

        # When they are different choose the one that is true, if none is true than there is no face
        else:
            if bbs_face_bool:
                xmin = bbs[0][0]
                ymin = bbs[0][1]
                xmax = bbs[1][0]
                ymax = bbs[1][1]
                inf_bb = np.array([(xmin, ymin), (xmax, ymax)]).reshape((-1, 2))
                snout_crop = cc_target_img[ymin:ymax, xmin:xmax]
                return snout_crop, inf_bb, False, True, bbs_ff_conf
            elif haar_face_bool:
                xmin = haar_bbs[0][0]
                ymin = haar_bbs[0][1]
                xmax = haar_bbs[1][0]
                ymax = haar_bbs[1][1]
                inf_bb = np.array([(xmin, ymin), (xmax, ymax)]).reshape((-1, 2))
                snout_crop = cc_target_img[ymin:ymax, xmin:xmax]
                return snout_crop, inf_bb, False, True, haar_ff_conf
            else:
                ff_conf = (bbs_ff_conf + haar_ff_conf) / 2 if haar_face_bool else bbs_ff_conf
                return None, None, False, False, ff_conf

    def calc_iou(self, gt_bbox, pred_bbox):
        (x_topleft_gt, y_topleft_gt), (x_bottomright_gt, y_bottomright_gt) = gt_bbox.tolist()
        (x_topleft_p, y_topleft_p), (x_bottomright_p, y_bottomright_p) = pred_bbox.tolist()

        if x_topleft_gt > x_bottomright_gt or y_topleft_gt > y_bottomright_gt:
            raise AssertionError("Ground Truth Bounding Box is not correct")
        if x_topleft_p > x_bottomright_p or y_topleft_p > y_bottomright_p:
            raise AssertionError("Predicted Bounding Box is not correct",
                                 x_topleft_p,
                                 x_bottomright_p,
                                 y_topleft_p,
                                 y_bottomright_gt)

        # if the GT bbox and predcited BBox do not overlap then iou=0
        if x_bottomright_gt < x_topleft_p:
            # If bottom right of x-coordinate GT bbox is less than or above the top left
            # of x coordinate of the predicted BBox
            return 0.0
        if y_bottomright_gt < y_topleft_p:
            # If bottom right of y-coordinate GT bbox is less than or above the top left
            # of y coordinate of the predicted BBox
            return 0.0
        if x_topleft_gt > x_bottomright_p:
            # If bottom right of x-coordinate GT bbox is greater than or below the bottom
            # right of x coordinate of the predicted BBox
            return 0.0
        if y_topleft_gt > y_bottomright_p:
            # If bottom right of y-coordinate GT bbox is greater than or below the bottom
            # right of y coordinate of the predicted BBox
            return 0.0

        GT_bbox_area = (x_bottomright_gt - x_topleft_gt + 1) * (y_bottomright_gt - y_topleft_gt + 1)
        Pred_bbox_area = (x_bottomright_p - x_topleft_p + 1) * (y_bottomright_p - y_topleft_p + 1)

        x_top_left = np.max([x_topleft_gt, x_topleft_p])
        y_top_left = np.max([y_topleft_gt, y_topleft_p])
        x_bottom_right = np.min([x_bottomright_gt, x_bottomright_p])
        y_bottom_right = np.min([y_bottomright_gt, y_bottomright_p])

        intersection_area = (x_bottom_right - x_top_left + 1) * (y_bottom_right - y_top_left + 1)

        union_area = (GT_bbox_area + Pred_bbox_area - intersection_area)

        return intersection_area / union_area

    def do_cc_mobile_stage(self, cc_target_img):
        pred_cc_bb_full, cat_bool, inference_time = self.cc_mobile_stage.do_cc(target_img=cc_target_img)
        dk_bool = False if cat_bool is True else True
        if cat_bool:
            bbs_xmin = pred_cc_bb_full[0][0]
            bbs_ymin = pred_cc_bb_full[0][1]
            bbs_xmax = pred_cc_bb_full[1][0]
            bbs_ymax = pred_cc_bb_full[1][1]
            bbs_target_img = cc_target_img[bbs_ymin:bbs_ymax, bbs_xmin:bbs_xmax]
            return dk_bool, cat_bool, bbs_target_img, pred_cc_bb_full, inference_time
        else:
            return dk_bool, cat_bool, None, None, inference_time

    def do_eyes_stage(self, eye_target_img, cc_pred_bb, cc_target_img):
        snout_crop, bbs, inference_time = self.eyes_stage.do_eyes(cc_target_img, eye_target_img, cc_pred_bb)
        return snout_crop, bbs, inference_time

    def do_haar_stage(self, target_img, pred_cc_bb_full, cc_target_img):
        haar_bbs, haar_inference_time, haar_found_bool = self.haar_stage.haar_do(target_img=target_img,
                                                                                 cc_bbs=pred_cc_bb_full,
                                                                                 full_img=cc_target_img)
        pc_xmin = int(haar_bbs[0][0])
        pc_ymin = int(haar_bbs[0][1])
        pc_xmax = int(haar_bbs[1][0])
        pc_ymax = int(haar_bbs[1][1])
        snout_crop = cc_target_img[pc_ymin:pc_ymax, pc_xmin:pc_xmax].copy()

        return snout_crop, haar_bbs, haar_inference_time, haar_found_bool

    def do_ff_stage(self, snout_crop):
        face_bool, ff_conf, ff_inference_time = self.ff_stage.ff_do(target_img=snout_crop)
        dk_bool = False if face_bool is True else True
        return dk_bool, face_bool, ff_conf, ff_inference_time

    def do_pc_stage(self, pc_target_img):
        pred_class, pred_val, inference_time = self.pc_stage.pc_do(target_img=pc_target_img)
        return pred_class, pred_val, inference_time

    def input_text(self, img, text, text_pos, color):
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 2
        lineType = 3

        cv2.putText(img, text,
                    text_pos,
                    font,
                    fontScale,
                    color,
                    lineType)
        return img
