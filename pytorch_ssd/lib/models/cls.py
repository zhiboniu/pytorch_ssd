import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.layers import *

class CLS(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, base):
        super(CLS, self).__init__()
        self.base = nn.ModuleList(base)
        self.criterion = None

    def forward(self, x, phase='train', match_result=None, tb_writer=None):
        sources = list()
        loc = list()
        conf = list()

#         import pdb
#         pdb.set_trace()
        # apply vgg up to conv4_3 relu
        for k in range(len(self.base)):  # TODO make it configurable
            x = self.base[k](x)
        return x

