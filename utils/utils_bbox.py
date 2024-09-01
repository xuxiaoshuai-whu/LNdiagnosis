import numpy as np
from .utils import _resize

def unmold_mask(mask, bbox, image_shape):
    y1, x1, y2, x2 = bbox
    mask = _resize(mask, (y2 - y1, x2 - x1))

    full_mask = np.zeros(image_shape[:2], dtype=np.float32)
    full_mask[y1:y2, x1:x2] = mask
    return full_mask

def norm_boxes(boxes, shape):
    h, w    = shape
    scale   = np.array([h - 1, w - 1, h - 1, w - 1])
    shift   = np.array([0, 0, 1, 1])
    return np.divide((boxes - shift), scale).astype(np.float32)


def denorm_boxes(boxes, shape):
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.around(np.multiply(boxes, scale) + shift).astype(np.int32)
    
def postprocess(detections, mrcnn_mask, image_shape, input_shape, window):

    zero_ix = np.where(detections[:, 4] == 0)[0]
    N       = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]   
    if N == 0:
        return None, None, None, None, None

    box_thre    = detections[:N, :4]
    class_thre  = detections[:N, 5]
    class_ids   = detections[:N, 4].astype(np.int32)

    file_path = "file.txt"  

    masks       = mrcnn_mask[np.arange(N), :, :, class_ids]

    class_ids   = class_ids - 1

    wy1, wx1, wy2, wx2 = norm_boxes(window, input_shape[:2])
    wh = wy2 - wy1
    ww = wx2 - wx1


    shift = np.array([wy1, wx1, wy1, wx1])
    scale = np.array([wh, ww, wh, ww])
    box_thre = np.divide(box_thre - shift, scale)
    box_thre = denorm_boxes(box_thre, image_shape[:2])

    exclude_ix = np.where((box_thre[:, 2] - box_thre[:, 0]) * (box_thre[:, 3] - box_thre[:, 1]) <= 0)[0]
    if exclude_ix.shape[0] > 0:
        box_thre    = np.delete(box_thre, exclude_ix, axis=0)
        class_thre  = np.delete(class_thre, exclude_ix, axis=0)
        class_ids   = np.delete(class_ids, exclude_ix, axis=0)
        masks       = np.delete(masks, exclude_ix, axis=0)
        N           = class_ids.shape[0]

    masks_sigmoid = []
    for i in range(N):
        full_mask = unmold_mask(masks[i], box_thre[i], image_shape)
        masks_sigmoid.append(full_mask)
    masks_sigmoid = np.stack(masks_sigmoid, axis=-1) if masks_sigmoid else np.empty(image_shape[:2] + (0,))


    masks_arg       = np.argmax(masks_sigmoid, axis=-1)

    masks_sigmoid   = masks_sigmoid > 0.5
    
    return box_thre, class_thre, class_ids, masks_arg, masks_sigmoid