import torch
from torch import nn as nn
from torch.autograd import Variable

from .backbone import *
from .ssd import SSD
from .ssd_more import SSD_MORE
from .ssd_coco import SSD_COCO
from .fpn import FPN
from .fssd import FSSD
from .rfb import RFB
from .cls import CLS
from .se_resnet import *
from .se_resnext import *
from lib.layers import PriorBoxSSD

bases_list = ['backbone']
ssds_list = ['SSD', 'FSSD', 'FPN', 'SSD_COCO', 'SSD_MORE', 'RFB']
priors_list = ['PriorBoxSSD']


def create(n, lst, **kwargs):
    if n not in lst:
        raise Exception("unkown type {}, possible: {}".format(n, str(lst)))
    return eval('{}(**kwargs)'.format(n))


def model_factory(phase, cfg):
    prior = create(cfg.MODEL.PRIOR_TYPE, priors_list, cfg=cfg)
    cfg.MODEL.NUM_PRIOR = prior.num_priors
    base = create(cfg.MODEL.BASE, bases_list)
    model = create(cfg.MODEL.SSD_TYPE, ssds_list, phase=phase, cfg=cfg, base=base)
    layer_dims = get_layer_dims(model, cfg.MODEL.IMAGE_SIZE)
    priors = prior.forward(layer_dims)
    return model, priors, layer_dims

def clsmodel_factory(phase, cfg):
#     base = create(cfg.MODEL.BASE, bases_list)
#     model = vgg16_cls(base, cfg.DATASET.NUM_CLASSES, phase)
    model = se_resnext_50()
    return model

def get_layer_dims(model, image_size):
    def forward_hook(self, input, output):
        """input: type tuple, output: type Variable"""
        # print('{} forward\t input: {}\t output: {}\t output_norm: {}'.format(
        #     self.__class__.__name__, input[0].size(), output.datasets.size(), output.datasets.norm()))
        dims.append([input[0].size()[2], input[0].size()[3]])  # h, w

    dims = []
    handles = []
    for idx, layer in enumerate(model.loc.children()):  # loc...
        if isinstance(layer, nn.Conv2d):
            hook = layer.register_forward_hook(forward_hook)
            handles.append(hook)

    input_size = (1, 3, image_size[0], image_size[1])
    model.eval()  # fix bn bugs
    model(Variable(torch.randn(input_size)), phase='eval')
    [item.remove() for item in handles]
    return dims
