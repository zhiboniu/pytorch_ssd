"""Adapted from:
    Licensed under The MIT License [see LICENSE for details]
"""

from __future__ import print_function
import sys
import re
sys.path[0] = '/train/execute/slagcar/slagcar_bce/pytorch_cs'
selfpath = [x for x in sys.modules if re.match('lib.',x)]
for x in selfpath: 
    del sys.modules[x]
print('selfpathcache:',[x for x in sys.modules if re.match('lib.',x)])

import argparse
import os
import os.path as osp

import numpy as np
import torch
from torch.autograd import Variable

from lib.utils.config import cfg, merge_cfg_from_file
from lib.datasets import dataset_factory
from lib.utils.utils import setup_cuda, setup_folder


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Evaluation')
parser.add_argument('--cfg_name', default='slagcls', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--job_group', default='base', type=str,
                    help='Directory for saving checkpoint models')
parser.add_argument('--trained_model', default=None, type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='./results/debug', type=str,
                    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--num_workers', default=8, type=int,
                    help='cpu workers for datasets processing')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--devices', default='0,1', type=str,
                    help='GPU visiable')
parser.add_argument('--net_gpus', default=[0], type=list,
                    help='GPU to use for net forward')
parser.add_argument('--loss_gpu', default=0, type=list,
                    help='GPU to use for loss calculation')
parser.add_argument('--cleanup', default=True, type=str2bool,
                    help='Cleanup and remove results files following eval')
parser.add_argument('--tocaffe', default=False, type=bool,
                    help='trans to caffe or not')

args = parser.parse_args()

def slagcarfilter(images):
    modle_dir = '/train/execute/slagcar/slagcar_bce/pytorch_cs/modelsave/'
    args.trained_model = 'slagcls_10000.pth'
    model_dir = osp.join(modle_dir, args.trained_model)
    print('eval model:{}'.format(model_dir))

#     setup_cuda(cfg, args.cuda, args.devices)

    np.set_printoptions(precision=3, suppress=True, edgeitems=4)

    images = Variable(images.cuda(), volatile=True)
    # load net
#     net = clsmodel_factory(phase='eval', cfg=cfg)
#     net.load_state_dict(torch.load(model_dir)['state_dict'].state_dict())
    net = torch.load(model_dir)['state_dict']

    if args.cuda:
        net = torch.nn.DataParallel(net)
        net = net.cuda()
    net.eval()
    
    preds = net(images)

    return preds