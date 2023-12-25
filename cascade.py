import numpy as np
import os
import cv2
import time
import csv
import sys
import gc
import pytz
import traceback
import logging
from datetime import datetime
from collections import deque
from threading import Thread
from multiprocessing import Process
import xml.etree.ElementTree as ET

from model_stages import PCStage, FFStage, EyeStage, HaarStage, CCMobileNetStage
from camera_class import Camera
from telegram_bot import NodeBot

cat_cam_py = os.getenv('CAT_PREY_ANALYZER_PATH')

# We configure the logging
logger = logging.getLogger("cat_logger")
logger.setLevel(logging.INFO)

stdout_handler = logging.StreamHandler(stream=sys.stdout)
file_handler = logging.FileHandler(filename='/tmp/cat_logger.log')
dbg_file_handler = logging.FileHandler(filename='/tmp/cat_logger-dbg.log')
stdout_handler.setLevel(logging.INFO)
file_handler.setLevel(logging.INFO)
dbg_file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

stdout_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
dbg_file_handler.setFormatter(formatter)

logger.addHandler(stdout_handler)
logger.addHandler(file_handler)
logger.addHandler(dbg_file_handler)


class SequentialCascadeFeeder:
    def __init__(self):
        self.log_dir = os.path.join(os.getcwd(), 'log')
        logger.info('Log Dir:' + str(self.log_dir))
        self.event_nr = 0
        self.base_cascade = Cascade()
        self.DEFAULT_FPS_OFFSET = 2
        self.QUEUE_MAX_THRESHOLD = 50
        self.fps_offset = self.DEFAULT_FPS_OFFSET
        self.MAX_PROCESSES = 5
        self.EVENT_FLAG = False
        self.event_objects = []
        self.patience_counter = 0
        self.PATIENCE_FLAG = False
        self.FACE_FOUND_FLAG = False
        self.event_reset_threshold = 6
        self.event_reset_counter = 0
        self.cat_counter = 0
        self.cat_counter_threshold = 6
        self.cumulus_points = 0
        self.cumulus_prey_threshold = -10
        self.cumulus_no_prey_threshold = 2.9603
        self.prey_val_hard_threshold = 0.6
        self.face_counter = 0
        self.PREY_FLAG = None
        self.NO_PREY_FLAG = None
        self.queues_cumuli_in_event = []
        self.bot = NodeBot()
        self.processing_pool = []

        self.main_deque = deque()

        self.camera = Camera(fps=5)

    def reset_cumuli_et_al(self):
        self.EVENT_FLAG = False
        self.patience_counter = 0
        self.PATIENCE_FLAG = False
        self.FACE_FOUND_FLAG = False
        self.cumulus_points = 0
        self.fps_offset = self.DEFAULT_FPS_OFFSET
        self.event_reset_counter = 0
        self.cat_counter = 0
        self.face_counter = 0
        self.PREY_FLAG = None
        self.NO_PREY_FLAG = None
        self.cumulus_points = 0

        # Close the node_letin flag
        self.bot.node_let_in_flag = False

        self.event_objects.clear()
        self.queues_cumuli_in_event.clear()
        self.main_deque.clear()

        # terminate processes when pool too large
        if len(self.processing_pool) >= self.MAX_PROCESSES:
            logger.debug('Terminating oldest processes Len:' + str(len(self.processing_pool)))
            for p in self.processing_pool[0:int(len(self.processing_pool) / 2)]:
                p.terminate()
            logger.debug('Now processes Len:' + str(len(self.processing_pool)))

    def log_event_to_csv(self, event_obj, queues_cumuli_in_event, event_nr):
        csv_name = 'event_log.csv'
        file_exists = os.path.isfile(os.path.join(self.log_dir, csv_name))
        with open(os.path.join(self.log_dir, csv_name), mode='a') as csv_file:
            headers = ['Event', 'Img_Name', 'Done_Time', 'Queue', 'Cumuli', 'CC_Cat_Bool', 'CC_Time', 'CR_Class',
                       'CR_Val', 'CR_Time', 'BBS_Time', 'HAAR_Time', 'FF_BBS_Bool', 'FF_BBS_Val', 'FF_BBS_Time',
                       'Face_Bool', 'PC_Class', 'PC_Val', 'PC_Time', 'Total_Time']
            writer = csv.DictWriter(csv_file, delimiter=',', lineterminator='\n', fieldnames=headers)
            if not file_exists:
                writer.writeheader()

            for i, img_obj in enumerate(event_obj):
                writer.writerow(
                    {'Event': event_nr, 'Img_Name': img_obj.img_name, 'Done_Time': queues_cumuli_in_event[i][2],
                     'Queue': queues_cumuli_in_event[i][0],
                     'Cumuli': queues_cumuli_in_event[i][1], 'CC_Cat_Bool': img_obj.cc_cat_bool,
                     'CC_Time': img_obj.cc_inference_time, 'CR_Class': img_obj.cr_class,
                     'CR_Val': img_obj.cr_val, 'CR_Time': img_obj.cr_inference_time,
                     'BBS_Time': img_obj.bbs_inference_time,
                     'HAAR_Time': img_obj.haar_inference_time, 'FF_BBS_Bool': img_obj.ff_bbs_bool,
                     'FF_BBS_Val': img_obj.ff_bbs_val, 'FF_BBS_Time': img_obj.ff_bbs_inference_time,
                     'Face_Bool': img_obj.face_bool,
                     'PC_Class': img_obj.pc_prey_class, 'PC_Val': img_obj.pc_prey_val,
                     'PC_Time': img_obj.pc_inference_time, 'Total_Time': img_obj.total_inference_time})

    def queue_worker(self):
        logger.info('Working the Queue with len:' + str(len(self.main_deque)))
        start_time = time.time()
        # Feed the latest image in the Queue through the cascade
        cascade_obj = self.feed(target_img=self.main_deque[self.fps_offset][1], img_name=self.main_deque[self.fps_offset][0])[1]
        logger.debug('Runtime:' + str(time.time() - start_time))
        done_timestamp = datetime.now(pytz.timezone('Europe/Zurich')).strftime("%Y_%m_%d_%H-%M-%S.%f")
        logger.debug('Timestamp at Done Runtime:' + str(done_timestamp))

        overhead = datetime.strptime(done_timestamp, "%Y_%m_%d_%H-%M-%S.%f") - datetime.strptime(
            self.main_deque[self.fps_offset][0], "%Y_%m_%d_%H-%M-%S.%f")
        logger.debug('Overhead:' + str(overhead.total_seconds()))

        # Add this such that the bot has some info
        self.bot.node_queue_info = len(self.main_deque)
        self.bot.node_live_img = self.main_deque[self.fps_offset][1]
        self.bot.node_over_head_info = overhead.total_seconds()

        # Always delete the left part
        for i in range(self.fps_offset + 1):
            self.main_deque.popleft()

        if cascade_obj.cc_cat_bool:
            # We are inside an event => add event_obj to list
            self.EVENT_FLAG = True
            self.event_nr = self.get_event_nr()
            self.event_objects.append(cascade_obj)
            # Send a message on Telegram to ask what to do
            self.cat_counter += 1
            if self.cat_counter >= self.cat_counter_threshold:
                msg_thread = Process(target=self.send_cat_detected_message, args=(self.bot.node_live_img, 0,),
                                     daemon=True)
                msg_thread.start()

            # Last cat pic for bot
            self.bot.node_last_casc_img = cascade_obj.output_img

            self.fps_offset = 0
            # If face found add the cumulus points
            if cascade_obj.face_bool:
                self.face_counter += 1
                self.cumulus_points += (50 - int(round(100 * cascade_obj.pc_prey_val)))
                self.FACE_FOUND_FLAG = True

            logger.info('CUMULUS:' + str(self.cumulus_points))
            self.queues_cumuli_in_event.append((len(self.main_deque), self.cumulus_points, done_timestamp))

            # Check the cumuli points and set flags if necessary
            if self.face_counter > 0 and self.PATIENCE_FLAG:
                if self.cumulus_points / self.face_counter > self.cumulus_no_prey_threshold:
                    self.NO_PREY_FLAG = True
                    logger.info('NO PREY DETECTED... YOU CLEAN...')
                    p = Process(target=self.send_no_prey_message,
                                args=(self.event_objects, self.cumulus_points / self.face_counter,), daemon=True)
                    p.start()
                    self.processing_pool.append(p)
                    # self.log_event_to_csv(event_obj=self.event_objects, queues_cumuli_in_event=self.queues_cumuli_in_event, event_nr=self.event_nr)
                    self.reset_cumuli_et_al()
                elif self.cumulus_points / self.face_counter < self.cumulus_prey_threshold:
                    self.PREY_FLAG = True
                    logger.info('IT IS A PREY!!!!!')
                    p = Process(target=self.send_prey_message,
                                args=(self.event_objects, self.cumulus_points / self.face_counter,), daemon=True)
                    p.start()
                    self.processing_pool.append(p)
                    # self.log_event_to_csv(event_obj=self.event_objects, queues_cumuli_in_event=self.queues_cumuli_in_event, event_nr=self.event_nr)
                    self.reset_cumuli_et_al()
                else:
                    self.NO_PREY_FLAG = False
                    self.PREY_FLAG = False

            # Cat was found => still belongs to event => acts as dk state
            self.event_reset_counter = 0
            self.cat_counter = 0

        # No cat detected => reset event_counters if necessary
        else:
            logger.info('NO CAT FOUND!')
            self.event_reset_counter += 1
            if self.event_reset_counter >= self.event_reset_threshold:
                # If was True => event now over => clear queue
                if self.EVENT_FLAG == True:
                    logger.info('CLEARED QUEUE BECAUSE EVENT OVER WITHOUT CONCLUSION...')
                    # TODO QUICK FIX
                    if self.face_counter == 0:
                        self.face_counter = 1
                    p = Process(target=self.send_dk_message,
                                args=(self.event_objects, self.cumulus_points / self.face_counter,), daemon=True)
                    p.start()
                    self.processing_pool.append(p)
                    # self.log_event_to_csv(event_obj=self.event_objects, queues_cumuli_in_event=self.queues_cumuli_in_event, event_nr=self.event_nr)
                self.reset_cumuli_et_al()

        if self.EVENT_FLAG and self.FACE_FOUND_FLAG:
            self.patience_counter += 1
        if self.patience_counter > 2:
            self.PATIENCE_FLAG = True
        if self.face_counter > 1:
            self.PATIENCE_FLAG = True

    def single_debug(self):
        start_time = time.time()
        target_img_name = 'dummy_img.jpg'
        target_img = cv2.imread(os.path.join(cat_cam_py, 'readme_images/lenna_casc_Node1_001557_02_2020_05_24_09-49-35.jpg'))
        cascade_obj = self.feed(target_img=target_img, img_name=target_img_name)[1]
        logger.debug('Runtime:' + str(time.time() - start_time))
        return cascade_obj

    def queue_handler(self):
        # Do this to force run all networks s.t. the network inference time stabilizes
        self.single_debug()

        camera_thread = Thread(target=self.camera.fill_queue, args=(self.main_deque,), daemon=True)
        camera_thread.start()

        while True:
            if len(self.main_deque) > self.QUEUE_MAX_THRESHOLD:
                self.main_deque.clear()
                self.reset_cumuli_et_al()
                # Clean up garbage
                gc.collect()
                logger.info('DELETING QUEUE BECAUSE OVERLOADED!')
                self.bot.send_text(message='Running Hot... had to kill Queue!')

            elif len(self.main_deque) > self.DEFAULT_FPS_OFFSET:
                self.queue_worker()

            else:
                logger.info('Nothing to work with => Queue_length:', len(self.main_deque))
                time.sleep(0.25)

            # Check if user force opens the door
            if self.bot.node_let_in_flag:
                # We do super simple stuff here. The actual unlock of the door is handled in NodeBot class
                self.reset_cumuli_et_al()

    def stop_threads(self):
        # We stop the camera thread
        self.camera.stop_thread()

    def send_prey_message(self, event_objects, cumuli):
        prey_vals = [x.pc_prey_val for x in event_objects]
        max_prey_index = prey_vals.index(max(filter(lambda x: x is not None, prey_vals)))

        event_str = ''
        face_events = [x for x in event_objects if x.face_bool]
        for f_event in face_events:
            logger.info('****************')
            logger.info('Img_Name:' + str(f_event.img_name))
            logger.info('PC_Val:' + str('%.2f' % f_event.pc_prey_val))
            logger.info('****************')
            event_str += '\n' + f_event.img_name + ' => PC_Val: ' + str('%.2f' % f_event.pc_prey_val)

        sender_img = event_objects[max_prey_index].output_img
        caption = 'Cumuli: ' + str(cumuli) + ' => PREY IN DA HOUSE!' + ' 🐁🐁🐁' + event_str
        self.bot.send_img(img=sender_img, caption=caption)
        return

    def send_no_prey_message(self, event_objects, cumuli):
        prey_vals = [x.pc_prey_val for x in event_objects]
        min_prey_index = prey_vals.index(min(filter(lambda x: x is not None, prey_vals)))

        event_str = ''
        face_events = [x for x in event_objects if x.face_bool]
        for f_event in face_events:
            logger.info('****************')
            logger.info('Img_Name:' + str(f_event.img_name))
            logger.info('PC_Val:' + str('%.2f' % f_event.pc_prey_val))
            logger.info('****************')
            event_str += '\n' + f_event.img_name + ' => PC_Val: ' + str('%.2f' % f_event.pc_prey_val)

        sender_img = event_objects[min_prey_index].output_img
        caption = 'Cumuli: ' + str(cumuli) + ' => Cat is clean...' + ' 🐱' + event_str
        self.bot.send_img(img=sender_img, caption=caption)
        return

    def send_dk_message(self, event_objects, cumuli):
        event_str = ''
        face_events = [x for x in event_objects if x.face_bool]
        for f_event in face_events:
            logger.info('****************')
            logger.info('Img_Name:' + str(f_event.img_name))
            logger.info('PC_Val:' + str('%.2f' % f_event.pc_prey_val))
            logger.info('****************')
            event_str += '\n' + f_event.img_name + ' => PC_Val: ' + str('%.2f' % f_event.pc_prey_val)

        sender_img = face_events[0].output_img
        caption = 'Cumuli: ' + str(cumuli) + ' => Cant say for sure...' + ' 🤷‍♀️' + event_str + '\nMaybe use /letin?'
        self.bot.send_img(img=sender_img, caption=caption)
        return

    def send_cat_detected_message(self, live_img, cumuli):
        caption = 'Cumuli: ' + str(
            cumuli) + ' => Gato incoming! 🐱🐈🐱' + '\nMaybe use /letin, /unlock, /lock, /lockin or /lockout?'
        # sender_img = event_objects[-1].output_img
        self.bot.send_img(img=live_img, caption=caption);

    def get_event_nr(self):
        tree = ET.parse(os.path.join(self.log_dir, 'info.xml'))
        data = tree.getroot()
        imgNr = int(data.find('node').get('imgNr'))
        data.find('node').set('imgNr', str(int(imgNr) + 1))
        tree.write(os.path.join(self.log_dir, 'info.xml'))

        return imgNr

    def feed(self, target_img, img_name):
        target_event_obj = EventElement(img_name=img_name, cc_target_img=target_img)

        start_time = time.time()
        single_cascade = self.base_cascade.do_single_cascade(event_img_object=target_event_obj)
        single_cascade.total_inference_time = sum(filter(None, [
            single_cascade.cc_inference_time,
            single_cascade.cr_inference_time,
            single_cascade.bbs_inference_time,
            single_cascade.haar_inference_time,
            single_cascade.ff_bbs_inference_time,
            single_cascade.ff_haar_inference_time,
            single_cascade.pc_inference_time]))
        total_runtime = time.time() - start_time
        logger.debug('Total Runtime:' + str(total_runtime))

        return total_runtime, single_cascade


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

    def do_single_cascade(self, event_img_object):
        logger.info('Processing image: ' + str(event_img_object.img_name))
        cc_target_img = event_img_object.cc_target_img
        original_copy_img = cc_target_img.copy()

        # Do CC
        start_time = time.time()
        dk_bool, cat_bool, bbs_target_img, pred_cc_bb_full, cc_inference_time = self.do_cc_mobile_stage(
            cc_target_img=cc_target_img)
        logger.debug('CC_Do Time:' + str(time.time() - start_time))
        event_img_object.cc_cat_bool = cat_bool
        event_img_object.cc_pred_bb = pred_cc_bb_full
        event_img_object.bbs_target_img = bbs_target_img
        event_img_object.cc_inference_time = cc_inference_time

        if cat_bool and bbs_target_img.size != 0:
            logger.info('Cat Detected!')
            rec_img = self.cc_mobile_stage.draw_rectangle(img=original_copy_img, box=pred_cc_bb_full, color=(255, 0, 0),
                                                          text='CC_Pred')

            # Do HAAR
            haar_snout_crop, haar_bbs, haar_inference_time, haar_found_bool = self.do_haar_stage(
                target_img=bbs_target_img, pred_cc_bb_full=pred_cc_bb_full, cc_target_img=cc_target_img)
            rec_img = self.cc_mobile_stage.draw_rectangle(img=rec_img, box=haar_bbs, color=(0, 255, 255),
                                                          text='HAAR_Pred')

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
                logger.info('Face Detected!')

                # Do PC
                pred_class, pred_val, inference_time = self.do_pc_stage(pc_target_img=snout_crop)
                logger.debug('Prey Prediction: ' + str(pred_class))
                logger.debug('Pred_Val: ' + str('%.2f' % pred_val))
                pc_str = ' PC_Pred: ' + str(pred_class) + ' @ ' + str('%.2f' % pred_val)
                color = (0, 0, 255) if pred_class else (0, 255, 0)
                rec_img = self.input_text(img=rec_img, text=pc_str, text_pos=(15, 100), color=color)

                event_img_object.pc_prey_class = pred_class
                event_img_object.pc_prey_val = pred_val
                event_img_object.pc_inference_time = inference_time

            else:
                logger.info('No Face Found...')
                ff_str = 'No_Face'
                rec_img = self.input_text(img=rec_img, text=ff_str, text_pos=(15, 100), color=(255, 255, 0))

        else:
            logger.info('No Cat Found...')
            rec_img = self.input_text(img=original_copy_img, text='CC_Pred: NoCat', text_pos=(15, 100),
                                      color=(255, 255, 0))

        # Always save rec_img in event_img object
        event_img_object.output_img = rec_img
        return event_img_object

    def cc_haar_overlap(self, cc_bbs, haar_bbs):
        cc_area = abs(cc_bbs[0][0] - cc_bbs[1][0]) * abs(cc_bbs[0][1] - cc_bbs[1][1])
        haar_area = abs(haar_bbs[0][0] - haar_bbs[1][0]) * abs(haar_bbs[0][1] - haar_bbs[1][1])
        overlap = haar_area / cc_area
        logger.debug('Overlap: ' + str(overlap))
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

        if (x_topleft_gt > x_bottomright_gt) or (y_topleft_gt > y_bottomright_gt):
            raise AssertionError("Ground Truth Bounding Box is not correct")
        if (x_topleft_p > x_bottomright_p) or (y_topleft_p > y_bottomright_p):
            raise AssertionError("Predicted Bounding Box is not correct", x_topleft_p, x_bottomright_p, y_topleft_p,
                                 y_bottomright_gt)

        # if the GT bbox and predcited BBox do not overlap then iou=0
        if (
                x_bottomright_gt < x_topleft_p):  # If bottom right of x-coordinate  GT  bbox is less than or above the top left of x coordinate of  the predicted BBox
            return 0.0
        if (
                y_bottomright_gt < y_topleft_p):  # If bottom right of y-coordinate  GT  bbox is less than or above the top left of y coordinate of  the predicted BBox
            return 0.0
        if (
                x_topleft_gt > x_bottomright_p):  # If bottom right of x-coordinate  GT  bbox is greater than or below the bottom right  of x coordinate of  the predcited BBox
            return 0.0
        if (
                y_topleft_gt > y_bottomright_p):  # If bottom right of y-coordinate  GT  bbox is greater than or below the bottom right  of y coordinate of  the predcited BBox
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


if __name__ == '__main__':
    sq_cascade = SequentialCascadeFeeder()
    try:
        sq_cascade.queue_handler()
    except Exception as e:
        print("Something wrong happened... Message:", e)
        logger.exception("Something wrong happened... Restarting", e)
        traceback.print_exc()
        sys.exit(1)
