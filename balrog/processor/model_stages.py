import os
import pathlib
import sys
import time
from pathlib import Path
from typing import Tuple, Sequence

import cv2
import numpy as np
import tensorflow as tf
from cv2.typing import MatLike

from balrog.config import general_config
from balrog.processor.cv_helpers import resize_img_to_square
from balrog.types import Box
from balrog.utils import logger, get_resource_path

_tensorflow_models_path = os.getenv('BALROG_TENSORFLOW_PATH')
if (_tensorflow_models_path is None or
        len(_tensorflow_models_path) <= 0 or
        not pathlib.Path(_tensorflow_models_path).is_dir()):
    raise Exception("The BALROG_TENSOFLOW_PATH was not set, or points to an invalid location. Please check the asigned value")

sys.path.append(_tensorflow_models_path)

from object_detection.utils import label_map_util


_PC_model_file = 'models/Prey_Classifier/0.86_512_05_VGG16_ownData_FTfrom15_350_Epochs_2020_05_15_11_40_56.h5'
_FF_model_file = 'models/Face_Fur_Classifier/256_05_mobileNet_50_Epochs_2020_05_07_14_56_25.h5'
_EYE_model_file = 'models/Eye_Detector/trainwhole100_Epochs_2020_04_30_18_05_25.h5'
_HAAR_model_file = 'models/Haar_Classifier/haarcascade_frontalcatface_extended.xml'

_TF_OD_model_name = 'ssdlite_mobilenet_v2_coco_2018_05_09'
_TF_OD_frozen_model_filename = 'frozen_inference_graph.pb'
_TF_OD_labels_filename = 'data/mscoco_label_map.pbtxt'

_CR_model_file = 'models/Cat_Recognizer'


class CCMobileNetStage:
    def __init__(self):
        # ****Initialize TensorFlow model****
        # * Load the label map.
        # Label maps map indices to category names, so that when the convolution
        # network predicts `5`, we know that this corresponds to `airplane`.
        # Here we use internal utility functions, but anything that returns a
        # dictionary mapping integers to appropriate string labels would be fine

        # Path to frozen detection graph .pb file, which contains the model that is used
        # for object detection.
        frozen_model_file: Path = Path(f'{_tensorflow_models_path}/object_detection/'
                                       f'{_TF_OD_model_name}/{_TF_OD_frozen_model_filename}').resolve()

        # Path to label map file
        labels_file: Path = Path(f'{_tensorflow_models_path}/object_detection/{_TF_OD_labels_filename}').resolve()
        label_map = label_map_util.load_labelmap(str(labels_file))

        # Number of classes the object detector can identify
        num_classes = 90
        categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                    max_num_classes=num_classes,
                                                                    use_display_name=True
                                                                    )
        self.category_index = label_map_util.create_category_index(categories)

        # Config tensorflow
        config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=2,
                                          inter_op_parallelism_threads=general_config.max_frame_processor_threads * 2,
                                          allow_soft_placement=True,
                                          device_count={'CPU': 2})

        # * Load the Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.compat.v2.io.gfile.GFile(str(frozen_model_file), 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.compat.v1.Session(graph=detection_graph, config=config)

        # * Define input and output tensors (i.e. data) for the object detection classifier

        # Input tensor is the image
        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        # Output tensors are the detection boxes, scores, and classes
        # Each box represents a part of the image where a particular object was detected
        self.detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represents level of confidence for each of the objects.
        # The score is shown on the result image, together with the class label.
        self.detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

        # Number of objects detected
        self.num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        logger.info(f"TF Config: inter_threads = {tf.config.threading.get_inter_op_parallelism_threads()}")
        logger.info(f"TF Config: intra_threads = {tf.config.threading.get_intra_op_parallelism_threads()}")
        logger.debug('CNN is ready to go!')

    def do_cc(self, target_img: MatLike) -> Tuple[bool, Box, float]:
        colored_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
        resized_colored_img = resize_img_to_square(colored_img, 300, 1.0)

        pet_detected, pet_box, inference_time = self._pet_detector(target_img, resized_colored_img)
        return pet_detected, pet_box, inference_time

    # This function contains the code to detect a pet, determine if it's
    # inside or outside, and send a text to the user's phone.
    def _pet_detector(self, original_frame: MatLike, resized_frame: MatLike) -> Tuple[bool, Box, float]:
        frame_expanded = np.expand_dims(resized_frame, axis=0)

        # Perform the actual detection by running the model with the image as input
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: frame_expanded})
        end_time = time.time()
        inference_time = end_time - start_time

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


class HaarStage:
    def __init__(self):
        with get_resource_path(_HAAR_model_file) as model_file:
            self.face_cascade = cv2.CascadeClassifier(str(model_file).strip())

    def haar_do(self, sub_img: MatLike, full_img: MatLike, prev_box: Box) -> Tuple[bool, Box, float]:
        face_found, face_found_box, inference_time,  = self._haar_predict(sub_img)
        logger.debug('Haar_time: ' + str('%.2f' % inference_time))

        face_box = face_found_box[:]

        face_box[0][0] = max(face_found_box[0][0] + prev_box[0][0], 0)
        face_box[0][1] = max(face_found_box[0][1] + prev_box[0][1], 0)
        face_box[1][0] = min(face_found_box[1][0] + prev_box[0][0], full_img.shape[1])
        face_box[1][1] = min(face_found_box[1][1] + prev_box[0][1], full_img.shape[0])

        return face_found, face_box, inference_time

    def _haar_predict(self, input_img: MatLike) -> Tuple[bool, Box, float]:
        start_time = time.time()
        bw_image = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

        faces: Sequence[cv2.typing.Rect] = self.face_cascade.detectMultiScale(
            image=bw_image, scaleFactor=1.3, minNeighbors=1, minSize=(25, 25)
        )
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


class PCStage:
    def __init__(self):
        # Handle args
        with get_resource_path(_PC_model_file) as model_file:
            custom_objects = {'get_f1': PCStage._get_f1} if 'F1' in _PC_model_file else None
            self.pc_model = tf.keras.models.load_model(str(model_file),
                                                       custom_objects=custom_objects)

    @staticmethod
    def _get_f1(y_true, y_pred):  # taken from old keras source code
        import tensorflow.keras.backend as keras

        true_positives = keras.sum(keras.round(keras.clip(y_true * y_pred, 0, 1)))
        possible_positives = keras.sum(keras.round(keras.clip(y_true, 0, 1)))
        predicted_positives = keras.sum(keras.round(keras.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + keras.epsilon())
        recall = true_positives / (possible_positives + keras.epsilon())
        f1_val = 2 * (precision * recall) / (precision + recall + keras.epsilon())
        return f1_val

    def pc_do(self, target_img: MatLike) -> Tuple[bool, float, float]:
        size = 224
        preprocessed_img = resize_img_to_square(target_img, size, (1. / 255)).reshape((1, size, size, 3))

        start_time = time.time()
        class_pred = self.pc_model.predict(preprocessed_img)
        inference_time = time.time() - start_time

        prey_value = class_pred[0][0]

        return prey_value <= 0.5, prey_value, inference_time


class FFStage:
    def __init__(self):
        # Handle args
        with get_resource_path(_FF_model_file) as model_file:
            self.ff_model: tf.keras.Model = tf.keras.models.load_model(str(model_file))

    def face_fur_do(self, target_img: MatLike) -> Tuple[bool, float, float]:
        size = 224
        preprocessed_img = resize_img_to_square(target_img, size, (1. / 255)).reshape((1, size, size, 3))

        start_time = time.time()
        class_pred = self.ff_model.predict(preprocessed_img)
        inference_time = time.time() - start_time

        pred = class_pred[0][0]

        return pred <= 0.5, pred, inference_time


class EyeStage:
    TARGET_SIZE = 224

    def __init__(self):
        with get_resource_path(_EYE_model_file) as model_file:
            self.eye_model: tf.keras.Model = tf.keras.models.load_model(str(model_file))

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
        preprocessed_img, top, left = self._resize_img(image)
        inputs = (preprocessed_img.astype('float32') / 255).reshape((1, self.TARGET_SIZE, self.TARGET_SIZE, 3))
        start_time = time.time()
        pred_eyes = self.eye_model.predict(inputs)[0].reshape((-1, 2))
        inference_time = time.time() - start_time

        ratio_h = self.TARGET_SIZE / image.shape[0]
        ratio_w = self.TARGET_SIZE / image.shape[1]

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
        eyes_box = EyeStage._eyes_to_box(pred_eyes=eyes_coords, image=raw_image, face_box=in_box)

        pc_xmin = int(eyes_box[0][0])
        pc_ymin = int(eyes_box[0][1])
        pc_xmax = int(eyes_box[1][0])
        pc_ymax = int(eyes_box[1][1])
        eyes_img_crop = raw_image[pc_ymin:pc_ymax, pc_xmin:pc_xmax].copy()

        return eyes_img_crop, eyes_box, inference_time
