import torch.utils.data as data
from .det_dataset import detection_collate
from .voc0712 import VOCDetection, VOCAnnotationTransform, VOC_CLASSES
from .coco import COCODetection, COCOAnnotationTransform, get_label_map
from .selfdata import SelfDef_Detection, SelfDef_CLS, get_label_map
from .config import *
from lib.utils.augmentations import SSDAugmentation, CLSAugmentation
from lib.utils.rfn_augment import RFBAugmentation

dataset_map = {'VOC0712': VOCDetection,
               'COCO2014': COCODetection,
              'DEEPV': SelfDef_Detection,
              'CLS': SelfDef_CLS}

augmentation_map = {
    'SSD': SSDAugmentation,
    'RFB': RFBAugmentation,
    'CLS': CLSAugmentation
}


def dataset_factory(phase, cfg):
    det_dataset = dataset_map[cfg.DATASET.NAME]
    det_aug = augmentation_map[cfg.AUGMENTATION.NAME]
    if phase == 'train':
        dataset = det_dataset(cfg.DATASET.DATASET_DIR, cfg.DATASET.TRAIN_SETS,
                              det_aug(cfg, cfg.DATASET.IMAGE_SIZE, cfg.DATASET.PIXEL_MEANS))
        data_loader = data.DataLoader(dataset, batch_size=cfg.DATASET.TRAIN_BATCH_SIZE,
                                      num_workers=cfg.DATASET.NUM_WORKERS,
                                      shuffle=True, collate_fn=detection_collate,
                                      pin_memory=True, drop_last=True)
    elif phase == 'eval':
        dataset = det_dataset(cfg.DATASET.DATASET_DIR, cfg.DATASET.TEST_SETS,
                              det_aug(cfg, cfg.DATASET.IMAGE_SIZE, cfg.DATASET.PIXEL_MEANS,
                                              use_base=True))
        data_loader = data.DataLoader(dataset, batch_size=cfg.DATASET.EVAL_BATCH_SIZE,
                                      num_workers=cfg.DATASET.NUM_WORKERS, shuffle=False,
                                      collate_fn=detection_collate, pin_memory=True)#, drop_last=True)
    else:
        raise Exception("unsupported phase type")
    return data_loader
