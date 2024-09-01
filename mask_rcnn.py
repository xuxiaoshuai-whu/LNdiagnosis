import colorsys
import os
import time

import cv2
import numpy as np
from PIL import Image

from nets.mrcnn import get_predict_model
from utils.anchors import get_anchors
from utils.config import Config
from utils.utils import cvtColor, get_classes, resize_image
from utils.utils_bbox import postprocess


class MASK_RCNN(object):
    _defaults = {

        # "model_path"        : 'model_data/20210831_epoch048_loss0.998_val_loss0.950.h5',
        "model_path": 'logs/ep040-loss0.807-val_loss0.816.h5',
        "classes_path": 'model_data/ln_classes.txt',
 
        "confidence": 0.85,

        "nms_iou": 0.3,

        "IMAGE_MAX_DIM": 512,

        "RPN_ANCHOR_SCALES": [16, 32, 64, 128, 256]
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.num_classes += 1
        if self.num_classes <= 81:
            self.colors = np.array([[0, 0, 0], [0, 255, 0], [255, 0, 0], [156, 39, 176], [103, 58, 183],
                                    [100, 30, 60], [63, 81, 181], [33, 150, 243], [3, 169, 244], [0, 188, 212],
                                    [20, 55, 200], [0, 150, 136], [76, 175, 80], [139, 195, 74], [205, 220, 57],
                                    [70, 25, 100], [255, 235, 59], [255, 193, 7], [255, 152, 0], [255, 87, 34],
                                    [90, 155, 50], [121, 85, 72], [158, 158, 158], [96, 125, 139], [15, 67, 34],
                                    [98, 55, 20], [21, 82, 172], [58, 128, 255], [196, 125, 39], [75, 27, 134],
                                    [90, 125, 120], [121, 82, 7], [158, 58, 8], [96, 25, 9], [115, 7, 234],
                                    [8, 155, 220], [221, 25, 72], [188, 58, 158], [56, 175, 19], [215, 67, 64],
                                    [198, 75, 20], [62, 185, 22], [108, 70, 58], [160, 225, 39], [95, 60, 144],
                                    [78, 155, 120], [101, 25, 142], [48, 198, 28], [96, 225, 200], [150, 167, 134],
                                    [18, 185, 90], [21, 145, 172], [98, 68, 78], [196, 105, 19], [215, 67, 84],
                                    [130, 115, 170], [255, 0, 255], [255, 255, 0], [196, 185, 10], [95, 167, 234],
                                    [18, 25, 190], [0, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255],
                                    [155, 0, 0], [0, 155, 0], [0, 0, 155], [46, 22, 130], [255, 0, 155],
                                    [155, 0, 255], [255, 155, 0], [155, 255, 0], [0, 155, 255], [0, 255, 155],
                                    [18, 5, 40], [120, 120, 255], [255, 58, 30], [60, 45, 60], [75, 27, 244],
                                    [128, 25, 70]], dtype='uint8')
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        class InferenceConfig(Config):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            NUM_CLASSES = self.num_classes
            DETECTION_MIN_CONFIDENCE = self.confidence
            DETECTION_NMS_THRESHOLD = self.nms_iou

            RPN_ANCHOR_SCALES = self.RPN_ANCHOR_SCALES
            IMAGE_MAX_DIM = self.IMAGE_MAX_DIM

        self.config = InferenceConfig()
        self.config.display()
        self.generate()

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'


        self.model = get_predict_model(self.config)
        self.model.load_weights(self.model_path, by_name=True)


    def detect_image(self, image, img_name):
        image_shape = np.array(np.shape(image)[0:2])

        image = cvtColor(image)
        image_origin = np.array(image, np.uint8)

        image_data, image_metas, windows = resize_image([np.array(image)], self.config)

        anchors = np.expand_dims(get_anchors(self.config, image_data[0].shape), 0)
 
        detections, _, _, mrcnn_mask, _, _, _ = self.model.predict([image_data, image_metas, anchors], verbose=0)


        box_thre, class_thre, class_ids, masks_arg, masks_sigmoid = postprocess(
            detections[0], mrcnn_mask[0], image_shape, image_data[0].shape, windows[0]
        )

        if box_thre is None:
            return image

        masks_class = masks_sigmoid * (class_ids[None, None, :] + 1)
        masks_class = np.reshape(masks_class, [-1, np.shape(masks_sigmoid)[-1]])
        masks_class = np.reshape(masks_class[np.arange(np.shape(masks_class)[0]), np.reshape(masks_arg, [-1])],
                                 [image_shape[0], image_shape[1]])
  
        scale = 0.6
        thickness = int(max((image.size[0] + image.size[1]) // self.IMAGE_MAX_DIM, 1))
        font = cv2.FONT_HERSHEY_DUPLEX
        color_masks = self.colors[masks_class].astype('uint8')

        image_fused = cv2.addWeighted(color_masks, 0.4, image_origin, 0.6, gamma=0)

        normal_count = 0
        abnormal_count = 0
        lymph_data = {}
        lymph_list = []
        for i in range(np.shape(class_ids)[0]):

            top, left, bottom, right = np.array(box_thre[i, :], np.int32)

            if self.class_names[class_ids[i]] == "lymph_normal":
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)

            cv2.rectangle(image_fused, (left, top), (right, bottom), color, thickness)
            class_name = self.class_names[class_ids[i]]
            print(class_name, top, left, bottom, right)


            if class_name == "lymph_normal":
                normal_count += 1
            if class_name == "lymph_abnormal":
                abnormal_count += 1

            lymph = {
                "class_name": class_name,
                "top_left": "(" + str(top) + "," + str(left) + ")",
                "bottom_right": "(" + str(bottom) + "," + str(right) + ")",
            }

            # content = class_name + ":(" + str(top) + "," + str(left) + "),(" + str(bottom) + "," + str(
            #     right) + ")\n"  
            lymph_list.append(lymph)

            text_str = f'{class_name}: {class_thre[i]:.2f}'
            text_w, text_h = cv2.getTextSize(text_str, font, scale, 1)[0]
            # cv2.rectangle(image_fused, (left, top), (left + text_w, top + text_h + 5), color, -1)
            cv2.putText(image_fused, text_str, (left, bottom - 15), font, scale, (255, 255, 255), 1, cv2.LINE_AA)
        lymph_data['abnormal_count'] = abnormal_count
        lymph_data['normal_count'] = normal_count
        lymph_data['lymph_list'] = lymph_list
        image = Image.fromarray(np.uint8(image_fused))
        return image, lymph_data

    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])

        image = cvtColor(image)
 
        image_data, image_metas, windows = resize_image([np.array(image)], self.config)

        anchors = np.expand_dims(get_anchors(self.config, image_data[0].shape), 0)

        detections, _, _, mrcnn_mask, _, _, _ = self.model.predict([image_data, image_metas, anchors], verbose=0)

        box_thre, class_thre, class_ids, masks_arg, masks_sigmoid = postprocess(
            detections[0], mrcnn_mask[0], image_shape, image_data[0].shape, windows[0]
        )

        t1 = time.time()
        for _ in range(test_interval):

            detections, _, _, mrcnn_mask, _, _, _ = self.model.predict([image_data, image_metas, anchors], verbose=0)
            box_thre, class_thre, class_ids, masks_arg, masks_sigmoid = postprocess(
                detections[0], mrcnn_mask[0], image_shape, image_data[0].shape, windows[0]
            )
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def get_map_out(self, image):
        image_shape = np.array(np.shape(image)[0:2])

        image = cvtColor(image)
        image_data, image_metas, windows = resize_image([np.array(image)], self.config)
        anchors = np.expand_dims(get_anchors(self.config, image_data[0].shape), 0)
        detections, _, _, mrcnn_mask, _, _, _ = self.model.predict([image_data, image_metas, anchors], verbose=0)

        box_thre, class_thre, class_ids, masks_arg, masks_sigmoid = postprocess(
            detections[0], mrcnn_mask[0], image_shape, image_data[0].shape, windows[0]
        )

        outboxes = None
        if box_thre is not None:
            outboxes = np.zeros_like(box_thre)
            outboxes[:, [0, 2]] = box_thre[:, [1, 3]]
            outboxes[:, [1, 3]] = box_thre[:, [0, 2]]
        return outboxes, class_thre, class_ids, masks_arg, masks_sigmoid
