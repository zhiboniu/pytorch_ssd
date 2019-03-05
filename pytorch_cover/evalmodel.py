"""Adapted from:
    Licensed under The MIT License [see LICENSE for details]
"""

from __future__ import print_function
import argparse
import os.path as osp

import numpy as np
import torch
from torch.autograd import Variable

from lib.utils.config import cfg, merge_cfg_from_file
from lib.datasets import dataset_factory
from lib.models import model_factory
from lib.utils import eval_solver_factory
from lib.utils.utils import setup_cuda, setup_folder


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Evaluation')
parser.add_argument('--cfg_name', default='libraf', type=str,
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
                    help='GPU to use')
parser.add_argument('--net_gpus', default=[0,1], type=list,
                    help='GPU to use for net forward')
parser.add_argument('--loss_gpu', default=0, type=list,
                    help='GPU to use for loss calculation')
parser.add_argument('--cleanup', default=True, type=str2bool,
                    help='Cleanup and remove results files following eval')

args = parser.parse_args()

if __name__ == '__main__':
    tb_writer, cfg_path, snapshot_dir, log_dir = setup_folder(args, cfg, phase='eval')
    merge_cfg_from_file(cfg_path)
    cfg.DATASET.NUM_EVAL_PICS = 0

    # args.trained_model = './results/vgg16_ssd_coco_24.4.pth'
    # args.trained_model = './results/ssd300_mAP_77.43_v2.pth'
    args.trained_model = 'libraf_100000.pth'
    model_dir = osp.join(snapshot_dir, args.trained_model)

    setup_cuda(cfg, args.cuda, args.devices)

    np.set_printoptions(precision=3, suppress=True, edgeitems=4)
    loader = dataset_factory(phase='eval', cfg=cfg)

    # load net
    net, priors, _ = model_factory(phase='eval', cfg=cfg)
    # net.load_state_dict(torch.load(model_dir))
#     import pdb
#     pdb.set_trace()
#     net.load_state_dict(torch.load(model_dir)['state_dict'])
#     import pdb
#     pdb.set_trace()
    net = torch.load(model_dir)['state_dict']
    if args.cuda:
        net = torch.nn.DataParallel(net)
        net = net.cuda()
        priors = Variable(priors.cuda(), volatile=True)
    else:
        priors = Variable(priors)
    net.eval()

    print('test_type:', cfg.DATASET.TEST_SETS, 'test_model:', args.trained_model,
          'device_id:', cfg.GENERAL.CUDA_VISIBLE_DEVICES)

    eval_solver = eval_solver_factory(loader, cfg)
    res, mAPs = eval_solver.validate(net, priors, tb_writer=tb_writer)
    print('final mAP', mAPs)
    eval_solver.eval_pr_curve()
