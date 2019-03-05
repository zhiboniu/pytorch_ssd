from .config import HOME
import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
from PIL import Image
import numpy as np
import _pickle as cPickle
import xml.etree.ElementTree as ET
from lib.utils.config import labelmap_list
# labelmap_list = ["flammable_cls", "explosive_cls", "corrosive_cls", "other_danger_cls"]
# labelmap_list = ["person_upper","front_face","side_face"]
# labelmap_list = ["person_upper"]
# labelmap = {
#     "risk_mark_triangle":0,
#     "dangerous_signs":0,
#     "flammable_liquid":0,
#     "flammable_cls":1,
#     "explosive_cls":2,
#     "corrosive_cls":3,
#     "other_danger_cls":4
# }


def get_label_map(label_file):
    label_map = {}
    labels = open(label_file, 'r')
    for line in labels:
        ids = line.split(',')
        label_map[int(ids[0])] = int(ids[1])
    return label_map

def labelmap_creat(labelmap_list):
    labelmap = {}
    for i,lable in enumerate(labelmap_list):
        labelmap[lable] = i
    print("labelmap dict created:{}".format(labelmap))
    return labelmap

class COCOAnnotationTransform(object):
    """Transforms a COCO annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    """
    def __init__(self):
        self.label_map = get_label_map(osp.join(COCO_ROOT, 'coco_labels.txt'))

    def __call__(self, target, width, height):
        """
        Args:
            target (dict): COCO target json annotation as a python dict
            height (int): height
            width (int): width
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class idx]
        """
        scale = np.array([width, height, width, height])
        res = []
        for obj in target:
            if 'bbox' in obj:
                bbox = obj['bbox']
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]
                label_idx = self.label_map[obj['category_id']] - 1
                final_box = list(np.array(bbox)/scale)
                final_box.append(label_idx)
                res += [final_box]  # [xmin, ymin, xmax, ymax, label_idx]
            else:
                print("no bbox problem!")

        return res  # [[xmin, ymin, xmax, ymax, label_idx], ... ]


class SelfDef_Detection(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        set_name (string): Name of the specific set of COCO images.
        transform (callable, optional): A function/transform that augments the
                                        raw images`
        target_transform (callable, optional): A function/transform that takes
        in the target (bbox) and transforms it.
    """

    def __init__(self, root, image_set='trainval', transform=None,
                 target_transform=None, dataset_name='Detection dataloader'):
#         sys.path.append(osp.join(root, COCO_API))
#         from pycocotools.coco import COCO
#         self.root = osp.join(root, IMAGES, image_set)
#         self.coco = COCO(osp.join(root, ANNOTATIONS,
#                                   INSTANCES_SET.format(image_set)))
#         self.ids = list(self.coco.imgToAnns.keys())
#         self.cache_pkl = os.path.join(Cache_Path, image_set+'_cache.pkl')
        self.image_sets = os.path.join(root,image_set[0])
#         print(root)
#         if os.path.exists(self.cache_pkl):
#             with open(self.cache_pkl,'rb') as cf:
#                 self.ids = cPickle.load(cf)
#                 self.data = cPickle.load(cf)
#                 self.anno = cPickle.load(cf)
#         else:
#             if MAKE_CACHE:
#                 pass
#             else:
#                 with open(self.image_set,'r') as rf:
#                     self.ids = rf.readlines()
        with open(self.image_sets,'r') as rf:
            self.ids = rf.readlines()
        
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self.labelmap = labelmap_creat(labelmap_list)
        

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target).
                   target is the object returned by ``coco.loadAnns``.
        """
        im, gt, extras = self.pull_item(index)
        return im, gt, extras

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index, tb_writer=None):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target, height, width).
                   target is the object returned by ``coco.loadAnns``.
        """
        img_path,xml_path = self.ids[index].strip().split(' ')
        assert osp.exists(img_path), 'Image does not exist: {}'.format(img_path)
        assert osp.exists(xml_path), 'Image does not exist: {}'.format(xml_path)
        target = self.parsexml(xml_path)
        img = cv2.imread(img_path)
        height, width, _ = img.shape
        if self.target_transform is not None:
            target = self.target_transform(target, width, height)
#         if self.transform is not None:
#             target = np.array(target)
#             print("target shape:{}".format(target.shape))
#             img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
#             # to rgb
#             img = img[:, :, (2, 1, 0)]

#             target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
            
        if self.transform is not None:
            target = np.array(target)
#             print("target shape:{}".format(target.shape))
            if target.size == 0:
                img, boxes, labels = self.transform(img, None, None, tb_writer)  # target remains as is
                boxes = [[0, 0, 0, 0]]
                labels = [-1]
            else:
                img, boxes, labels = self.transform(img, target[:, :4], target[:, 4], tb_writer)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
            #img = img[:, :, (2, 1, 0)]  # to rgb
#         print(img_path)
#         print(target)
        return torch.from_numpy(img).permute(2, 0, 1), target, (height, width)

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            cv2 img
        '''
        img_path = self.ids[index].split(' ')[0]
        return cv2.imread(img_path, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        xml_path = self.ids[index].strip().split(' ')[1]
        target = self.parsexml(xml_path)
        return target
    
    def parsexml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        tinylist = []
        tiny_arr = np.array(tinylist)
        for subitem in root.findall('size'):
            width = float(subitem.find('width').text)
            height = float(subitem.find('height').text)
        for subitem in root.findall('object'):
            name = subitem.find('name').text
            if name not in labelmap_list:
                continue
            for items in subitem.findall('bndbox'):
                xmin = (float(items.find('xmin').text)/width)
                ymin = (float(items.find('ymin').text)/height)
                xmax = (float(items.find('xmax').text)/width)
                ymax = (float(items.find('ymax').text)/height)
            label = self.labelmap[name]
            sublist = [xmin, ymin, xmax, ymax, label]
            tinylist.append(sublist)
#             print(tiny_arr.shape,tiny_arr)
        return tinylist
    

class SelfDef_CLS(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        set_name (string): Name of the specific set of COCO images.
        transform (callable, optional): A function/transform that augments the
                                        raw images`
        target_transform (callable, optional): A function/transform that takes
        in the target (bbox) and transforms it.
    """

    def __init__(self, root, image_set='train.txt', transform=None,
                 target_transform=None, dataset_name='classification multilable'):
        self.image_sets = os.path.join(root,image_set[0])

        with open(self.image_sets,'r') as rf:
            self.ids = rf.readlines()
        
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self.labelmap = labelmap_creat(labelmap_list)
        

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target).
                   target is the object returned by ``coco.loadAnns``.
        """
        im, gt, extras = self.pull_item(index)
        return im, gt, extras

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index, tb_writer=None):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target, height, width).
                   target is the object returned by ``coco.loadAnns``.
        """
        img_path = self.ids[index].strip().split(' ')[0]
        target = [int(x) for x in self.ids[index].strip().split(' ')[1:]]
        assert osp.exists(img_path), 'Image does not exist: {}'.format(img_path)
        oimg = Image.open(img_path)
#         import pdb
#         pdb.set_trace()
        width,oheight = oimg.size
        height = max(min(width,oheight),oheight/2)
        img =  oimg.crop([0,0,width,height])
#         print("image size:{}".format(img.size))
        if self.target_transform is not None:
            print('target transform not implemented...')
            assert False
#             target = self.target_transform(target, width, height)
#         if self.transform is not None:
#             target = np.array(target)
#             print("target shape:{}".format(target.shape))
#             img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
#             # to rgb
#             img = img[:, :, (2, 1, 0)]

#             target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
            
        if self.transform is not None:
            target = np.array(target)
            
            img = self.transform(img)
#             img = img[:, :, (2, 1, 0)]  # to bgr
#         print(img_path)
#         print(target)
#         print(img.shape)
#         print('tatget:{}'.format(target))
        return img, target, (height, width)
