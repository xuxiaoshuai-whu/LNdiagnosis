import datetime
import os
import warnings

import tensorflow as tf
from pycocotools.coco import COCO
from tensorflow.keras.callbacks import (EarlyStopping, LearningRateScheduler,
                                        TensorBoard)
from tensorflow.keras.optimizers import SGD, Adam

from nets.mrcnn import get_train_model
from nets.mrcnn_training import get_lr_scheduler
from utils.anchors import compute_backbone_shapes, generate_pyramid_anchors
from utils.augmentations import Augmentation
from utils.callbacks import LossHistory, ModelCheckpoint
from utils.config import Config
from utils.dataloader import COCODetection
from utils.utils import get_classes, get_coco_label_map

warnings.filterwarnings("ignore")

if __name__ == "__main__":

    train_gpu       = [0,]

    classes_path    = 'model_data/ln_classes.txt'   

    model_path = "logs/ep030-loss0.837-val_loss0.835.h5"

    IMAGE_MAX_DIM       = 512

    RPN_ANCHOR_SCALES   = [16,32, 64, 128, 256]
    Init_Epoch      = 30
    Epoch           = 45
    batch_size      = 1

    Init_lr             = 1e-4
    Min_lr              = Init_lr * 0.01

    optimizer_type      = "adam"
    momentum            = 0.9
    weight_decay        = 0

    lr_decay_type       = "cos"

    save_period         = 5

    save_dir            = 'logs'

    num_workers         = 1
    
    train_image_path        = "datasets/ln/image"
    train_annotation_path   = "datasets/ln/json/train_annotations.json"
    val_image_path          = "datasets/ln/image"
    val_annotation_path     = "datasets/ln/json/val_annotations.json"

    os.environ["CUDA_VISIBLE_DEVICES"]  = ','.join(str(x) for x in train_gpu)
    ngpus_per_node                      = len(train_gpu)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        
    if ngpus_per_node > 1:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = None
    print('Number of devices: {}'.format(ngpus_per_node))
    class_names, num_classes = get_classes(classes_path)
    num_classes = num_classes + 1
    
    class TrainConfig(Config):
        GPU_COUNT                   = ngpus_per_node
        IMAGES_PER_GPU              = batch_size // ngpus_per_node
        NUM_CLASSES                 = num_classes
        WEIGHT_DECAY                = weight_decay
        
        RPN_ANCHOR_SCALES           = RPN_ANCHOR_SCALES
        IMAGE_MAX_DIM               = IMAGE_MAX_DIM

    config = TrainConfig()
    config.display()

    backbone_shapes = compute_backbone_shapes(config, config.IMAGE_SHAPE)
    anchors         = generate_pyramid_anchors(config.RPN_ANCHOR_SCALES, config.RPN_ANCHOR_RATIOS, backbone_shapes, config.BACKBONE_STRIDES, config.RPN_ANCHOR_STRIDE)
    
    if ngpus_per_node > 1:
        with strategy.scope():
            model  = get_train_model(config)
            if model_path != "":
                print('Load weights {}.'.format(model_path))
                model.load_weights(model_path, by_name=True, skip_mismatch=True)
    else:
        model  = get_train_model(config)
        if model_path != "":
            print('Load weights {}.'.format(model_path))
            model.load_weights(model_path, by_name=True, skip_mismatch=True)
        
    train_coco  = COCO(train_annotation_path)
    val_coco    = COCO(val_annotation_path)

    num_train   = len(list(train_coco.imgToAnns.keys()))
    print(num_train)
    num_val     = len(list(val_coco.imgToAnns.keys()))

    wanted_step = 1.5e4 if optimizer_type == "sgd" else 0.5e4
    total_step  = num_train // batch_size * Epoch
    if total_step <= wanted_step:
        if num_train // batch_size == 0:
            raise ValueError('数据集过小，无法进行训练')
        wanted_epoch = wanted_step // (num_train // batch_size) + 1
        print("\n\033[1;33;44m[Warning] 使用%s优化器时，建议将训练总步长设置到%d以上。\033[0m"%(optimizer_type, wanted_step))
        print("\033[1;33;44m[Warning] 本次运行的总训练数据量为%d，Unfreeze_batch_size为%d，共训练%d个Epoch，计算出总训练步长为%d。\033[0m"%(num_train, batch_size, Epoch, total_step))
        print("\033[1;33;44m[Warning] 由于总训练步长为%d，小于建议总步长%d，建议设置总世代为%d。\033[0m"%(total_step, wanted_step, wanted_epoch))

    COCO_LABEL_MAP  = get_coco_label_map(train_coco, class_names)

    if True:
        nbs             = 16
        lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 1e-1
        lr_limit_min    = 3e-5 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        optimizer = {
            'adam'  : Adam(lr = Init_lr_fit, beta_1 = momentum),
            'sgd'   : SGD(lr = Init_lr_fit, momentum = momentum, nesterov=True)
        }[optimizer_type]
        if ngpus_per_node > 1:
            with strategy.scope():
                model.compile(optimizer=optimizer, loss={
                    'rpn_class_loss'    : lambda y_true, y_pred: y_pred,
                    'rpn_bbox_loss'     : lambda y_true, y_pred: y_pred,
                    'mrcnn_class_loss'  : lambda y_true, y_pred: y_pred,
                    'mrcnn_bbox_loss'   : lambda y_true, y_pred: y_pred,
                    'mrcnn_mask_loss'   : lambda y_true, y_pred: y_pred
                })
        else:
            model.compile(optimizer=optimizer, loss={
                'rpn_class_loss'    : lambda y_true, y_pred: y_pred,
                'rpn_bbox_loss'     : lambda y_true, y_pred: y_pred,
                'mrcnn_class_loss'  : lambda y_true, y_pred: y_pred,
                'mrcnn_bbox_loss'   : lambda y_true, y_pred: y_pred,
                'mrcnn_mask_loss'   : lambda y_true, y_pred: y_pred
            })


        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, Epoch)
        
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size

        train_dataloader    = COCODetection(train_image_path, train_coco, num_classes, anchors, batch_size, config, COCO_LABEL_MAP, Augmentation(config.IMAGE_SHAPE))
        val_dataloader      = COCODetection(val_image_path, train_coco, num_classes, anchors, batch_size, config, COCO_LABEL_MAP, Augmentation(config.IMAGE_SHAPE))

        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
        logging         = TensorBoard(log_dir)
        loss_history    = LossHistory(log_dir)
        checkpoint      = ModelCheckpoint(os.path.join(save_dir, "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5"), 
                                monitor = 'val_loss', save_weights_only = True, save_best_only = False, period = save_period)
        checkpoint_last = ModelCheckpoint(os.path.join(save_dir, "last_epoch_weights.h5"), 
                               monitor = 'val_loss', save_weights_only = True, save_best_only = False, period = 1)
        checkpoint_best = ModelCheckpoint(os.path.join(save_dir, "best_epoch_weights.h5"), 
                                monitor = 'val_loss', save_weights_only = True, save_best_only = True, period = 1)
        #early_stopping  = EarlyStopping(monitor='val_loss', min_delta = 0, patience = 10, verbose = 1)
        lr_scheduler    = LearningRateScheduler(lr_scheduler_func, verbose = 1)
        callbacks       = [logging, loss_history, checkpoint, checkpoint_last, checkpoint_best, lr_scheduler]

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit(
            x                   = train_dataloader,
            steps_per_epoch     = epoch_step,
            validation_data     = val_dataloader,
            validation_steps    = epoch_step_val,
            epochs              = Epoch,
            initial_epoch       = Init_Epoch,
            use_multiprocessing = True if num_workers > 1 else False,
            workers             = num_workers,
            callbacks           = callbacks
        )
