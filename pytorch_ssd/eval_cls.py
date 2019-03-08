"""Adapted from:
    Licensed under The MIT License [see LICENSE for details]
"""

from __future__ import print_function
import argparse
import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from lib.utils.config import cfg, merge_cfg_from_file
from lib.datasets import dataset_factory
from lib.models import model_factory, clsmodel_factory
from lib.utils import eval_solver_factory
from lib.utils.utils import setup_cuda, setup_folder

from pytorch2caffe_v2 import ConvertModel
import sys


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Evaluation')
parser.add_argument('--cfg_name', default='slagcover', type=str,
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

if __name__ == '__main__':
    tb_writer, cfg_path, snapshot_dir, log_dir = setup_folder(args, cfg, phase='eval')
    merge_cfg_from_file(cfg_path)
    cfg.DATASET.NUM_EVAL_PICS = 0
    savedir = os.path.join(os.getcwd(),'savework',args.cfg_name)
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # args.trained_model = './results/vgg16_ssd_coco_24.4.pth'
    # args.trained_model = './results/ssd300_mAP_77.43_v2.pth'
    args.trained_model = 'slagcover_20000.pth'
    model_dir = osp.join(snapshot_dir, args.trained_model)
    #workdir = os.getcwd()
    #model_dir = os.path.join(workdir, 'modelsave/slagcover_11000.pth')
    print('eval model:{}'.format(model_dir))

    setup_cuda(cfg, args.cuda, args.devices)

    np.set_printoptions(precision=3, suppress=True, edgeitems=4)
    loader = dataset_factory(phase='eval', cfg=cfg)

    # load net
#     net = clsmodel_factory(phase='eval', cfg=cfg)
#     net.load_state_dict(torch.load(model_dir)['state_dict'].state_dict())
    net = torch.load(model_dir)['state_dict']
#     import pdb
#     pdb.set_trace()
    if args.cuda:
        net = torch.nn.DataParallel(net)
        net = net.cuda()
    net.eval()
    

    if args.tocaffe:
        from pytorch2caffe_v2.ConvertModel import ConvertModel_caffe
        sys.path.insert(0,'/opt/caffe-ssd/caffe/python')
        import caffe

        resize_h, resize_w = cfg.DATASET.IMAGE_SIZE
        InputShape = [1, 3, resize_h, resize_w]
        """  Convert!  """
        print('Converting...')

        text_net, binary_weights = ConvertModel_caffe(net, InputShape, softmax=False)

        model_path = os.path.join(log_dir,'caffemodel')
        model_name = args.trained_model.strip('.pth')
        prototxtfile = os.path.join(model_path, model_name+'.prototxt')
        caffemodelfile = os.path.join(model_path, model_name+'.caffemodel')
        
        import google.protobuf.text_format
        with open(prototxtfile, 'w') as f:
            f.write(google.protobuf.text_format.MessageToString(text_net))
        with open(caffemodelfile, 'w') as f:
            f.write(binary_weights.SerializeToString())
        exit()
        
        
    print('test_type:', cfg.DATASET.TEST_SETS, 'test_model:', args.trained_model,
          'device_id:', cfg.GENERAL.CUDA_VISIBLE_DEVICES)

    eval_solver = eval_solver_factory(loader, cfg)
    mAPs = eval_solver.validate(net, tb_writer=tb_writer)
    print('final mAP', mAPs)
    eval_solver.eval_pr_curve(savedir)
#     eval_solver.visfalse(savedir)
    print("ALL DONE!")
