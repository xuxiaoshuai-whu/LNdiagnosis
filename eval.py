import os
import os.path as osp

import tensorflow as tf
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from mask_rcnn import MASK_RCNN
from utils.utils import get_classes, get_coco_label_map
from utils.utils_map import Make_json, prep_metrics

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

if __name__ == '__main__':

    map_mode        = 0

    classes_path    = 'model_data/ln_classes.txt'   

    Image_dir       = "datasets/coco/JPEGImages"
    Json_path       = "datasets/coco/Jsons/test_annotations.json"

    map_out_path    = 'map_out'

    test_coco       = COCO(Json_path)
    class_names, _  = get_classes(classes_path)
    COCO_LABEL_MAP  = get_coco_label_map(test_coco, class_names)
    
    ids         = list(test_coco.imgToAnns.keys())

    if not osp.exists(map_out_path):
        os.makedirs(map_out_path)
        
    if map_mode == 0 or map_mode == 1:
        print("Load model.")
        yolact      = MASK_RCNN(confidence = 0.05, nms_iou = 0.5)
        print("Load model done.")
        
        print("Get predict result.")
        make_json   = Make_json(map_out_path, COCO_LABEL_MAP)
        for i, id in enumerate(tqdm(ids)):
            image_path  = osp.join(Image_dir, test_coco.loadImgs(id)[0]['file_name'])
            image       = Image.open(image_path)

            box_thre, class_thre, class_ids, masks_arg, masks_sigmoid = yolact.get_map_out(image)
            if box_thre is None:
                continue
            prep_metrics(box_thre, class_thre, class_ids, masks_sigmoid, id, make_json)
        make_json.dump()
        print(f'\nJson files dumped, saved in: \'eval_results/\', start evaluting.')

    if map_mode == 0 or map_mode == 2:
        bbox_dets = test_coco.loadRes(osp.join(map_out_path, "bbox_detections.json"))
        mask_dets = test_coco.loadRes(osp.join(map_out_path, "mask_detections.json"))

        print('\nEvaluating BBoxes:')
        bbox_eval = COCOeval(test_coco, bbox_dets, 'bbox')
        bbox_eval.evaluate()
        bbox_eval.accumulate()
        bbox_eval.summarize()

        print('\nEvaluating Masks:')
        bbox_eval = COCOeval(test_coco, mask_dets, 'segm')
        bbox_eval.evaluate()
        bbox_eval.accumulate()
        bbox_eval.summarize()
