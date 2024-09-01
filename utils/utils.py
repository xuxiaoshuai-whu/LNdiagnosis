import warnings
from distutils.version import LooseVersion

import numpy as np
import scipy
import skimage
import skimage.transform


def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 


def _resize(image, output_shape, order=1, mode='constant', cval=0, clip=True,
           preserve_range=False, anti_aliasing=False, anti_aliasing_sigma=None):
    if LooseVersion(skimage.__version__) >= LooseVersion("0.14"):
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range, anti_aliasing=anti_aliasing,
            anti_aliasing_sigma=anti_aliasing_sigma)
    else:
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range)


def letterbox_image(image, max_dim=None):

    image_dtype = image.dtype
    h, w        = image.shape[:2]
    scale       = 1


    if max_dim:
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max
  
    if scale != 1:
        image = _resize(image, (round(h * scale), round(w * scale)), preserve_range=True)

    h, w        = image.shape[:2]
    top_pad     = (max_dim - h) // 2
    bottom_pad  = max_dim - h - top_pad
    left_pad    = (max_dim - w) // 2
    right_pad   = max_dim - w - left_pad

    padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0,0)]
    image   = np.pad(image, padding, mode='constant', constant_values=0)
    window  = (top_pad, left_pad, h + top_pad, w + left_pad)

    crop    = None
    return image.astype(image_dtype), window, scale, padding, crop

def resize_image(images, config):
    image_datas = []
    image_metas = []
    windows     = []

    for image in images:
        image_data, window, scale, padding, crop = letterbox_image(image, max_dim=config.IMAGE_MAX_DIM)
        image_data = preprocess_input(image_data)
        image_meta = compose_image_meta(0, image.shape, image_data.shape, window, scale, np.zeros([config.NUM_CLASSES], dtype=np.int32))
        
        image_datas.append(image_data)
        image_metas.append(image_meta)
        windows.append(window)
    image_datas = np.stack(image_datas)
    image_metas = np.stack(image_metas)
    windows     = np.stack(windows)
    return image_datas, image_metas, windows

def letterbox_mask(mask, scale, padding, crop=None):

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
    if crop is not None:
        y, x, h, w = crop
        mask = mask[y:y + h, x:x + w]
    else:
        mask = np.pad(mask, padding, mode='constant', constant_values=0)
    return mask


def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

def preprocess_input(image):
    mean    = np.array([123.7, 116.8, 103.9])
    image   = image.astype(np.float32) - mean
    return image

def compose_image_meta(image_id, original_image_shape, image_shape,
                       window, scale, active_class_ids):
    meta = np.array(
        [image_id] +                  # size=1
        list(original_image_shape) +  # size=3
        list(image_shape) +           # size=3
        list(window) +                # size=4 (y1, x1, y2, x2) in image cooredinates
        [scale] +                     # size=1
        list(active_class_ids)        # size=num_classes
    )
    return meta

def get_coco_label_map(coco, class_names):
    COCO_LABEL_MAP = {}

    coco_cat_index_MAP = {}
    for index, cat in coco.cats.items():
        if cat['name'] == '_background_':
            continue
        coco_cat_index_MAP[cat['name']] = index

    for index, class_name in enumerate(class_names):
        COCO_LABEL_MAP[coco_cat_index_MAP[class_name]] = index + 1
    return COCO_LABEL_MAP
