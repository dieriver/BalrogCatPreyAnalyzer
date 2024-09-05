import copy as cpy
import time
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple, Sequence, Callable, Optional

import cv2
import kagglehub as hub
import numpy as np
import tensorflow as tf
from cv2.typing import MatLike

from balrog.processor.cv_helpers import resize_img_to_square
from balrog.processor.model import (CCMobileExecutor, HaarModelExecutor, PCExecutor, FFExecutor, EyesExecutor,
                                    perform_cc_mobile_detection, perform_haar_detection, perform_pc_detection,
                                    perform_ff_detection, perform_eye_detection)
from balrog.types import Box
from balrog.utils import get_resource_path

_PC_model_file = 'models/Prey_Classifier/0.86_512_05_VGG16_ownData_FTfrom15_350_Epochs_2020_05_15_11_40_56.h5'
_FF_model_file = 'models/Face_Fur_Classifier/256_05_mobileNet_50_Epochs_2020_05_07_14_56_25.h5'
_EYE_model_file = 'models/Eye_Detector/trainwhole100_Epochs_2020_04_30_18_05_25.h5'
_HAAR_model_file = 'models/Haar_Classifier/haarcascade_frontalcatface_extended.xml'


class CascadeStage(ABC):
    @staticmethod
    @abstractmethod
    def init_executor(max_workers: int):
        pass

    @abstractmethod
    def shutdown(self):
        pass



class CCMobileNetStage(CascadeStage):
    worker_pool: ProcessPoolExecutor

    @staticmethod
    def init_executor(max_workers: int):
        # "TF2" way to dynamically load the ssd model from Kaggle Hub and easily "call" the model
        # model_files = hub.model_download("tensorflow/centernet-resnet/tensorFlow2/50v2-512x512")
        model_files = hub.model_download("tensorflow/ssd-mobilenet-v2/tensorFlow2/ssd-mobilenet-v2")
        executor = CCMobileExecutor(model_files, max_workers)
        CCMobileNetStage.worker_pool = ProcessPoolExecutor(max_workers=max_workers, initializer=executor.init)
        for _ in range(max_workers):
            CCMobileNetStage.worker_pool.submit(executor.force_init)

    def shutdown(self):
        self.worker_pool.shutdown(wait=False, cancel_futures=True)

    def do_cc(self, target_img: MatLike) -> Tuple[bool, Box, float]:
        img_copy = cpy.deepcopy(target_img)
        colored_img = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        resized_colored_img = resize_img_to_square(colored_img, 300)
        resized_colored_img = np.expand_dims(resized_colored_img, axis=0)

        pet_detected, pet_box, inference_time = self._pet_detector(img_copy, resized_colored_img)
        return pet_detected, pet_box, inference_time

    # This function contains the code to detect a pet, determine if it's
    # inside or outside, and send a text to the user's phone.
    def _pet_detector(self, original_frame: MatLike, resized_frame: MatLike) -> Tuple[bool, Box, float]:
        assert isinstance(resized_frame, MatLike)

        # Perform the actual detection by running the model with the image as input
        start_time = time.time()
        frame_tensor = tf.convert_to_tensor(resized_frame, dtype=tf.uint8)
        future_result = self.worker_pool.submit(perform_cc_mobile_detection, frame_tensor)
        detection_result = future_result.result()
        classes = detection_result['detection_classes'].numpy()
        boxes = detection_result['detection_boxes'].numpy()
        detections = detection_result['num_detections'].numpy()

        end_time = time.time()
        inference_time = end_time - start_time

        if detections == 0:
            # Nothing was detected:
            pet_detected = False
            pet_box = None
        else:
            # Check the class of the top detected object by looking at classes[0][0].
            # If the top detected object is a cat (17) or a dog (18) (or a teddy bear (88) for test purposes),
            # find its center coordinates by looking at the boxes[0][0] variable.
            # boxes[0][0] variable holds coordinates of detected objects as (ymin, xmin, ymax, xmax)
            xmin = int(boxes[0][0][1] * original_frame.shape[1])
            ymin = int(boxes[0][0][0] * original_frame.shape[0])
            xmax = int(boxes[0][0][3] * original_frame.shape[1])
            ymax = int(boxes[0][0][2] * original_frame.shape[0])
            pet_box: Box = np.array([(xmin, ymin), (xmax, ymax)]).reshape((-1, 2))

            # A pet was detected if the top detected object is of class 17 (cat) or 18 (dog)
            pet_detected = int(classes[0][0]) == 17 or int(classes[0][0]) == 18

        return pet_detected, pet_box, inference_time


class HaarStage(CascadeStage):
    worker_pool: ProcessPoolExecutor

    @staticmethod
    def init_executor(max_workers: int):
        with get_resource_path(_HAAR_model_file) as model_file:
            executor = HaarModelExecutor(str(model_file))
        HaarStage.worker_pool = ProcessPoolExecutor(max_workers=max_workers, initializer=executor.init)
        for _ in range(max_workers):
            # We submit a "dummy" task to each worker in the executor; this forces to call the "init" method
            # This action forces to fork the main process as soon as possible, leaving the workers lightweight
            HaarStage.worker_pool.submit(executor.force_init)

    def shutdown(self):
        self.worker_pool.shutdown(wait=False, cancel_futures=True)

    def haar_do(self, sub_img: MatLike, full_img: MatLike, prev_box: Box) -> Tuple[bool, Box, float]:
        img_copy = cpy.deepcopy(sub_img)
        face_found, face_found_box, inference_time,  = self._haar_predict(img_copy)

        face_box = face_found_box[:]

        face_box[0][0] = max(face_found_box[0][0] + prev_box[0][0], 0)
        face_box[0][1] = max(face_found_box[0][1] + prev_box[0][1], 0)
        face_box[1][0] = min(face_found_box[1][0] + prev_box[0][0], full_img.shape[1])
        face_box[1][1] = min(face_found_box[1][1] + prev_box[0][1], full_img.shape[0])

        return face_found, face_box, inference_time

    def _haar_predict(self, input_img: MatLike) -> Tuple[bool, Box, float]:
        start_time = time.time()
        bw_image = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

        if bw_image.size != 0:
            future_detections = self.worker_pool.submit(perform_haar_detection, bw_image)
            faces: Sequence[cv2.typing.Rect] = future_detections.result()
        else:
            # Something happened with the image; it has a dimension of 0
            faces = []

        inference_time = time.time() - start_time

        if len(faces) != 0:
            # Lambda computes the area: (rect[0], rect[1]) => (x,y) of top-left point of the rectangle
            # rect[2], rect[3] = width, height of the rectangle
            face = max(faces, key=lambda rect: abs(rect[0] - rect[2]) * abs(rect[1] - rect[3])).reshape(
                (-1, 2))
            face_box = face[:]
            face_box[0][0] = int(face[0][0] - face[1][0] * 0.2)
            face_box[0][1] = int(face[0][1] - face[1][1] * 0.4)
            face_box[1][0] = int(face[0][0] + face[1][0] * 1.2)
            face_box[1][1] = int(face[0][1] + face[1][1] * 1.6)
            face_found = True
        else:
            face_found = False
            face_box = np.array([(0, 0), (0, 0)]).reshape((-1, 2))

        return face_found, face_box, inference_time


def _apply_keras_model_on_image(pool: ProcessPoolExecutor,
                                detector: Callable[[tf.Tensor], Optional[np.ndarray]],
                                img: MatLike) -> Tuple[bool, float, float]:
    size = 224
    img_copy = cpy.deepcopy(img)
    preprocessed_img = resize_img_to_square(img_copy, size, normalize=True).reshape((1, size, size, 3))

    start_time = time.time()
    tensor_img = tf.convert_to_tensor(preprocessed_img)
    detect_future = pool.submit(detector, tensor_img)
    class_pred = detect_future.result()
    inference_time = time.time() - start_time
    if class_pred is None:
        return False, 0.0, inference_time
    else:
        prey_value = class_pred[0][0]
        return prey_value <= 0.5, prey_value, inference_time


def _get_f1(y_true, y_pred):  # taken from old keras source code
    import tensorflow.keras.backend as keras

    true_positives = keras.sum(keras.round(keras.clip(y_true * y_pred, 0, 1)))
    possible_positives = keras.sum(keras.round(keras.clip(y_true, 0, 1)))
    predicted_positives = keras.sum(keras.round(keras.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + keras.epsilon())
    recall = true_positives / (possible_positives + keras.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + keras.epsilon())
    return f1_val


class PCStage(CascadeStage):
    worker_pool: ProcessPoolExecutor

    @staticmethod
    def init_executor(max_workers: int):
                # Handle args
        with get_resource_path(_PC_model_file) as model_file:
            custom_objects = {'get_f1': _get_f1} if 'F1' in _PC_model_file else None
            executor = PCExecutor(str(model_file), custom_objects, max_workers)
        PCStage.worker_pool = ProcessPoolExecutor(max_workers=max_workers, initializer=executor.init)
        for _ in range(max_workers):
            # We submit a "dummy" task to each worker in the executor; this forces to call the "init" method
            # This action forces to fork the main process as soon as possible, leaving the workers lightweight
            HaarStage.worker_pool.submit(executor.force_init)

    def shutdown(self):
        self.worker_pool.shutdown(wait=False, cancel_futures=True)

    def pc_do(self, target_img: MatLike) -> Tuple[bool, float, float]:
        return _apply_keras_model_on_image(self.worker_pool, perform_pc_detection, target_img)


class FFStage(CascadeStage):
    worker_pool: ProcessPoolExecutor

    @staticmethod
    def init_executor(max_workers: int):
        with get_resource_path(_FF_model_file) as model_file:
            executor = FFExecutor(str(model_file), max_workers)
        PCStage.worker_pool = ProcessPoolExecutor(max_workers=max_workers, initializer=executor.init)
        for _ in range(max_workers):
            # We submit a "dummy" task to each worker in the executor; this forces to call the "init" method
            # This action forces to fork the main process as soon as possible, leaving the workers lightweight
            HaarStage.worker_pool.submit(executor.force_init)

    def shutdown(self):
        self.worker_pool.shutdown(wait=False, cancel_futures=True)

    def face_fur_do(self, target_img: MatLike) -> Tuple[bool, float, float]:
        return _apply_keras_model_on_image(self.worker_pool, perform_ff_detection, target_img)


class EyesStage(CascadeStage):
    TARGET_SIZE = 224
    worker_pool: ProcessPoolExecutor

    @staticmethod
    def init_executor(max_workers: int):
        with get_resource_path(_EYE_model_file) as model_file:
            executor = EyesExecutor(str(model_file), max_workers)
        PCStage.worker_pool = ProcessPoolExecutor(max_workers=max_workers, initializer=executor.init)
        for _ in range(max_workers):
            # We submit a "dummy" task to each worker in the executor; this forces to call the "init" method
            # This action forces to fork the main process as soon as possible, leaving the workers lightweight
            HaarStage.worker_pool.submit(executor.force_init)

    def shutdown(self):
        self.worker_pool.shutdown(wait=False, cancel_futures=True)

    def _resize_img(self, img_resize: MatLike):
        old_size = img_resize.shape[:2]  # old_size is in (height, width) format
        ratio = float(self.TARGET_SIZE) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        # new_size should be in (width, height) format
        img_resize = cv2.resize(img_resize, (new_size[1], new_size[0]))
        delta_w = self.TARGET_SIZE - new_size[1]
        delta_h = self.TARGET_SIZE - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        img_resize = cv2.copyMakeBorder(img_resize, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return img_resize, top, left

    def _eye_full_prediction(self, image: MatLike, face_box: Box) -> Tuple[Box, float]:
        img_copy = cpy.deepcopy(image)
        preprocessed_img, top, left = self._resize_img(img_copy)
        inputs = (preprocessed_img.astype('float32') / 255).reshape((1, self.TARGET_SIZE, self.TARGET_SIZE, 3))
        start_time = time.time()
        predict_future = self.worker_pool.submit(perform_eye_detection, inputs)
        predict_result = predict_future.result()
        inference_time = time.time() - start_time

        if predict_result is None:
            return False, inference_time

        pred_eyes = predict_result[0].reshape((-1, 2))
        ratio_h = self.TARGET_SIZE / img_copy.shape[0]
        ratio_w = self.TARGET_SIZE / img_copy.shape[1]

        pred_eyes[0][0] = int(pred_eyes[0][0] / ratio_w) + face_box[0][0] - left
        pred_eyes[0][1] = int(pred_eyes[0][1] / ratio_h) + face_box[0][1] - top
        pred_eyes[1][0] = int(pred_eyes[1][0] / ratio_w) + face_box[0][0] - left
        pred_eyes[1][1] = int(pred_eyes[1][1] / ratio_h) + face_box[0][1] - top

        return pred_eyes, inference_time

    @staticmethod
    def _eyes_to_box(pred_eyes: Box, image: MatLike, face_box: Box) -> Box:
        x_mid = int((pred_eyes[0][0] + pred_eyes[1][0]) / 2)
        y_mid = int((pred_eyes[0][1] + pred_eyes[1][1]) / 2)

        cc_bb_x_diff = abs(face_box[0][0] - face_box[1][0])
        cc_bb_y_diff = abs(face_box[0][1] - face_box[1][1])
        cc_bb_diff = max(cc_bb_x_diff, cc_bb_y_diff)

        top_margin = cc_bb_diff / 4
        bottom_margin = cc_bb_diff / 3
        side_margin = cc_bb_diff / 4

        xmin = max(int(x_mid - side_margin), 0)
        ymin = max(int(y_mid - top_margin), 0)
        xmax = min(int(x_mid + side_margin), image.shape[1])
        ymax = min(int(y_mid + bottom_margin), image.shape[0])

        return np.array([(xmin, ymin), (xmax, ymax)]).reshape((-1, 2))

    def do_eyes(self, in_image: MatLike, in_box: Box, raw_image: MatLike) -> Tuple[MatLike, Box, float]:
        eyes_coords, inference_time = self._eye_full_prediction(image=in_image, face_box=in_box)
        eyes_box = EyesStage._eyes_to_box(pred_eyes=eyes_coords, image=raw_image, face_box=in_box)

        pc_xmin = int(eyes_box[0][0])
        pc_ymin = int(eyes_box[0][1])
        pc_xmax = int(eyes_box[1][0])
        pc_ymax = int(eyes_box[1][1])
        eyes_img_crop = cpy.deepcopy(raw_image[pc_ymin:pc_ymax, pc_xmin:pc_xmax])

        return eyes_img_crop, eyes_box, inference_time


def init_executors(max_open_cv_workers: int) -> None:
    HaarStage.init_executor(max_open_cv_workers)
    CCMobileNetStage.init_executor(max_open_cv_workers)
    PCStage.init_executor(max_open_cv_workers)
    FFStage.init_executor(max_open_cv_workers)
    EyesStage.init_executor(max_open_cv_workers)
