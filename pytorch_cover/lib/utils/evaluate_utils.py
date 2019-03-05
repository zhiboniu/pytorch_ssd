import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from lib.datasets.deepv_eval import get_output_dir, evaluate_detections, labelmap
from lib.layers import DetectOut
from lib.utils import visualize_utils
from lib.utils.utils import Timer
import subprocess
import stat
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

class EvalBase(object):
    def __init__(self, data_loader, cfg):
        self.detector = DetectOut(cfg)
        self.data_loader = data_loader
        self.dataset = self.data_loader.dataset
        self.name = self.dataset.name
        self.cfg = cfg
        self.results = None  # dict for voc and list for coco
        self.image_sets = self.dataset.image_sets
        self.preds = []
        self.evalfile = None

    def reset_results(self):
        raise NotImplementedError

    def convert_ssd_result(self, det, img_idx):
        """
        :param det:
        :param img_idx:
        :return: [xmin, ymin, xmax, ymax, score, image, cls, (cocoid)]
        """
        raise NotImplementedError

    def post_proc(self, det, img_idx, id):
        raise NotImplementedError

    def evaluate_stats(self, classes=None, tb_writer=None):
        return NotImplementedError

    # @profile
######################validate#######################
#依次将文件送给模型做预测。并整理输出结果，存入result中，然后调用评估函数
#取数据：data_loader中依次取数据
#loc, conf = net(images, phase='eval')输出结果：（batchsize,cls,200,4）、（batchsize,cls,200,1）
#整理输出结果：convert_ssd_result
#存入result中：post_proc
#调用评估函数：evaluate_stats
#####################################################
    def validate(self, net, priors, use_cuda=True, tb_writer=None):
        print('start evaluation')
        priors = priors.cuda(self.cfg.GENERAL.NET_CPUS[0])
        self.reset_results()
        self.preds = []
        img_idx = 0
        _t = {'im_detect': Timer(), 'misc': Timer()}
        _t['misc'].tic()
        for batch_idx, (images, targets, extra) in enumerate(self.data_loader):
            if batch_idx % 25 == 0:
                print('processed image', img_idx)
            if use_cuda:
                images = Variable(images.cuda(), volatile=True)
                extra = extra.cuda()
            else:
                images = Variable(images, volatile=True)

            _t['im_detect'].tic()
            loc, conf = net(images, phase='eval')
            # image, cls, #box, [score, xmin, ymin, xmax, ymax]
            detections = self.detector(loc, conf, priors)
            _t['im_detect'].toc(average=False)

            det = detections.data
            
            batchsize = det.shape[0]
            eval_det = self.convert_eval_result(det, img_idx)
            self.det2list(eval_det, img_idx, batchsize)
            
            h = extra[:, 0].unsqueeze(-1).unsqueeze(-1)
            w = extra[:, 1].unsqueeze(-1).unsqueeze(-1)
            det[:, :, :, 1] *= w  # xmin
            det[:, :, :, 3] *= w  # xmax
            det[:, :, :, 2] *= h  # ymin
            det[:, :, :, 4] *= h  # ymax
            
            det, id = self.convert_ssd_result(det, img_idx)
            # the format is now xmin, ymin, xmax, ymax, score, image, cls, (cocoid)
            if tb_writer is not None and tb_writer.cfg['show_test_image']:
                self.visualize_box(images, targets, h, w, det, img_idx, tb_writer)
            img_idx = self.post_proc(det, img_idx, id)

        _t['misc'].toc(average=False)
        # print(_t['im_detect'].total_time, _t['misc'].total_time)
        return self.evaluate_stats(None, tb_writer)

    def visualize_box(self, images, targets, h, w, det, img_idx, tb_writer):
        det_ = det.cpu().numpy()
        # det_ = det_[det_[:, 4] > 0.5]
        images = images.permute(0, 2, 3, 1)
        images = images.data.cpu().numpy()
        for idx in range(len(images)):
            img = images[idx].copy()
            img = img[:, :, (2, 1, 0)]
            img += np.array((104., 117., 123.), dtype=np.float32)  # TODO cfg

            det__ = det_[det_[:, 5] == idx]
            w_ = w[idx, :].cpu().numpy()
            h_ = h[idx, :].cpu().numpy()
            # w_r = 1000  # resize to 1000, h
            # h_r = w_r / w_ * h_
            w_r = 300  # resize to 1000, h
            h_r = 300
            det__[:, 0:4:2] = det__[:, 0:4:2] / w_ * w_r
            det__[:, 1:4:2] = det__[:, 1:4:2] / h_ * h_r

            t = targets[idx].numpy()  # ground truth
            t[:, 0:4:2] = t[:, 0:4:2] * w_r
            t[:, 1:4:2] = t[:, 1:4:2] * h_r
            t[:, 4] += 1  # TODO because of the traget transformer

            boxes = {'gt': t, 'pred': det__}
            tb_writer.cfg['steps'] = img_idx + idx
            if self.name == 'MS COCO':
                tb_writer.cfg['img_id'] = int(det__[0, 7]) if det__.size != 0 else 'no_detect'
            if self.name == 'VOC0712':
                tb_writer.cfg['img_id'] = int(det__[0, 5]) if det__.size != 0 else 'no_detect'
            tb_writer.cfg['thresh'] = 0.3
            visualize_utils.vis_img_box(img, boxes, (h_r, w_r), tb_writer)


class EvalVOC(EvalBase):
    def __init__(self, data_loader, cfg):
        super(EvalVOC, self).__init__(data_loader, cfg)
        self.test_set = self.image_sets
        if cfg.DATASET.NUM_EVAL_PICS > 0:
            raise Exception("not support voc")
        print('eval img num', len(self.dataset))

    def reset_results(self):
        self.results = [[[] for _ in range(len(self.dataset))]
                        for _ in range(self.cfg.MODEL.NUM_CLASSES)]

#################convert_eval_result##################
# concate det/id/cls。将预测结果det、图片id、cls分类整合到一起。
#det shape:(batchsize,cls+1,200(topk),1)
#id: arange(0, det.shape[0])
#cls: labelmap
#####################################################
    def convert_eval_result(self, det, img_idx):
        """
        :param det:
        :param img_idx:
        :return: [xmin, ymin, xmax, ymax, score, image, cls, (cocoid)]
        """
        # append image id and class to the detection results by manually broadcasting
        eval_id = torch.arange(0, det.shape[0]).unsqueeze(-1).expand(list(det.shape[:2])) \
            .unsqueeze(-1).expand(list(det.shape[:3])).unsqueeze(-1)
        cls = torch.arange(0, det.shape[1]).expand(list(det.shape[:2])).unsqueeze(-1) \
            .expand(list(det.shape[:3])).unsqueeze(-1)
        eval_det = torch.cat((det, eval_id, cls), 3)
        mymask = eval_det[:, :, :, 0].gt(0.).unsqueeze(-1).expand(eval_det.size())
        eval_det = torch.masked_select(eval_det, mymask).view(-1, 7)
        # xmin, ymin, xmax, ymax, cls, score, image
        eval_det = eval_det[:, [1, 2, 3, 4, 6, 0, 5]]
        return eval_det
    
#######################det2list######################
#每个pred格式为xmin, ymin, xmax, ymax, cls, score
#每张图对应预测结果为[pred,pred...]
#输入：eval_det矩阵
#输出：list格式[pred,pred...]
#####################################################
    def det2list(self, eval_det, img_idx, batchsize):
        eval_det = eval_det.cpu().numpy()
        
        for b_idx in range(batchsize):
            eval_det_ = eval_det[eval_det[:, 6] == b_idx]
#             if eval_det_.size == 0:
#                 continue
            self.preds.append(eval_det_.tolist())
        return
    
#######################dump_npz######################
#保存labelmap、img_xml、preds到npz文件里
#文件路径：self.cfg.DATASET.DATASET_DIR/'results'
#####################################################
    def dump_npz(self):
        labelmaplist = ['back_ground'] + labelmap
        outsavedir = os.path.join(self.cfg.DATASET.DATASET_DIR, 'results')
        if not os.path.exists(outsavedir):
            os.mkdir(outsavedir)
        print("dump preds to npzfile")
        npzfile = os.path.join(outsavedir,'predsfile.npz')
        with open(npzfile, 'wb') as scoreoutfile:
            pickle.dump(labelmaplist, scoreoutfile, True)
            pickle.dump(self.dataset.ids, scoreoutfile, True)
            pickle.dump(self.preds, scoreoutfile, True)
        
        self.evalfile = os.path.join(outsavedir,'eval.sh')
        with open(self.evalfile, 'w') as ef:
            print("python "+self.cfg.EVAL.EVAL_NET+" -f "+npzfile,file=ef)
        os.chmod(self.evalfile, stat.S_IRWXU)
        return

#######################dump_npz######################
#保存labelmap、img_xml、preds到npz文件里
#文件路径：self.cfg.DATASET.DATASET_DIR/'results'
#####################################################
    def eval_pr_curve(self):
        self.dump_npz()
        subprocess.call(self.evalfile, shell=True)
        return
#################convert_ssd_result##################
# concate det/id/cls。将预测结果det、图片id、cls分类整合到一起。
#det shape:(batchsize,cls+1,200(topk),1)
#id: arange(0, det.shape[0])
#cls: labelmap
#####################################################
    def convert_ssd_result(self, det, img_idx):
        """
        :param det:
        :param img_idx:
        :return: [xmin, ymin, xmax, ymax, score, image, cls, (cocoid)]
        """
        # append image id and class to the detection results by manually broadcasting
        id = torch.arange(0, det.shape[0]).unsqueeze(-1).expand(list(det.shape[:2])) \
            .unsqueeze(-1).expand(list(det.shape[:3])).unsqueeze(-1)
        cls = torch.arange(0, det.shape[1]).expand(list(det.shape[:2])).unsqueeze(-1) \
            .expand(list(det.shape[:3])).unsqueeze(-1)
        det = torch.cat((det, id, cls), 3)
        mymask = det[:, :, :, 0].gt(0.).unsqueeze(-1).expand(det.size())
        det = torch.masked_select(det, mymask).view(-1, 7)
        # xmin, ymin, xmax, ymax, score, image, cls
        det = det[:, [1, 2, 3, 4, 0, 5, 6]]
        return det, id

#####################post_proc########################
#依batchsize中顺序，按img_idx区分预测结果det。
#将det内容按label、img_idx存入result中。
######################################################
    def post_proc(self, det, img_idx, id):
        det = det.cpu().numpy()
        # det_tensors.append(det)
        
        for b_idx in range(id.shape[0]):
            det_ = det[det[:, 5] == b_idx]
            for cls_idx in range(1, id.shape[1]):  # skip bg class
                det__ = det_[det_[:, 6] == cls_idx]
                if det__.size == 0:
                    continue
                self.results[cls_idx][img_idx] = det__[:, 0:5].astype(np.float32, copy=False)
            img_idx += 1
        return img_idx
    
######################evaluate_stats######################
#将post_proc函数中得到的result格式的结果，存入pkl文件。并执行evaluate_detections程序。
#最终结果pkl文件保存在det_file文件
##########################################################
    def evaluate_stats(self, classes=None, tb_writer=None):
        output_dir = get_output_dir('ssd300_120000', os.path.basename(self.test_set.rstrip('.txt')))
        print("output_dir",output_dir)
        det_file = os.path.join(output_dir, 'detections.pkl')
        with open(det_file, 'wb') as f:
            pickle.dump(self.results, f, pickle.HIGHEST_PROTOCOL)
        print('Evaluating detections')
        res, mAP = evaluate_detections(self.results, output_dir, self.data_loader.dataset, test_set=self.test_set)
        if tb_writer is not None and tb_writer.cfg['show_pr_scalar']:
            visualize_utils.viz_pr_curve(res, tb_writer)
        return res, [mAP]


class EvalCOCO(EvalBase):
    def __init__(self, data_loader, cfg):
        super(EvalCOCO, self).__init__(data_loader, cfg)
        if cfg.DATASET.NUM_EVAL_PICS > 0:
            self.dataset.ids = self.dataset.ids[:cfg.DATASET.NUM_EVAL_PICS]
        print('eval img num', len(self.dataset))

    def reset_results(self):
        self.results = []

    def convert_ssd_result(self, det, img_idx):
        # append image id and class to the detection results by manually broadcasting
        id = torch.arange(0, det.shape[0]).unsqueeze(-1).expand(list(det.shape[:2])) \
            .unsqueeze(-1).expand(list(det.shape[:3])).unsqueeze(-1)
        cls = torch.arange(0, det.shape[1]).expand(list(det.shape[:2])).unsqueeze(-1) \
            .expand(list(det.shape[:3])).unsqueeze(-1)

        coco_id = torch.Tensor(self.dataset.ids[img_idx: img_idx + det.shape[0]])
        coco_id = coco_id.unsqueeze(-1).expand(list(det.shape[:2])) \
            .unsqueeze(-1).expand(list(det.shape[:3])).unsqueeze(-1)
        det = torch.cat((det, id, cls, coco_id), 3)

        mymask = det[:, :, :, 0].gt(0.).unsqueeze(-1).expand(det.size())
        det = torch.masked_select(det, mymask).view(-1, 8)
        # xmin, ymin, xmax, ymax, score, image, cls, cocoid
        det = det[:, [1, 2, 3, 4, 0, 5, 6, 7]]
        return det, id

    # @profile
    def post_proc(self, det, img_idx, id):
        # x1, y1, x2, y2, score, image, cls, cocoid
        det[:, 2] -= det[:, 0]  # w
        det[:, 3] -= det[:, 1]  # h
        # cocoid, x1, y1, x2, y2, score, cls
        det = det[:, [7, 0, 1, 2, 3, 4, 6]]
        det_ = det.cpu().numpy()
        # det__ = det_[det_[:, 5] > 0.5]
        self.results.append(det_)
        img_idx += id.shape[0]
        return img_idx

    def evaluate_stats(self, classes=None, tb_writer=None):
        from pycocotools.cocoeval import COCOeval
        res = np.vstack(self.results)
        for r in res:
            r[6] = self.dataset.target_transform.inver_map[r[6]]
        coco = self.dataset.cocos[0]['coco']
        coco_pred = coco.loadRes(res)
        cocoEval = COCOeval(coco, coco_pred, 'bbox')
        cocoEval.params.imgIds = self.dataset.ids
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        res = cocoEval.eval
        ap05 = res['precision'][0, :, :, 0, 2]
        map05 = np.mean(ap05[ap05 > -1])
        ap95 = res['precision'][:, :, :, 0, 2]
        map95 = np.mean(ap95[ap95 > -1])
        """
        # show precision of each class
        s = cocoEval.eval['precision'][0]
        t = s[:, :, 0, 2]
        m = np.mean(t[t>-1])
        rc = []
        for i in range(t.shape[1]):
            r = t[:, i]
            rc.append(np.mean(r[r>-1]))
        print(rc)
        """
        return res, [map05, map95]
    
class EvalCLS(object):
    def __init__(self, data_loader, cfg):
        self.detector = DetectOut(cfg)
        self.data_loader = data_loader
        self.dataset = self.data_loader.dataset
        self.name = self.dataset.name
        self.cfg = cfg
        self.results = None  # dict for voc and list for coco
        self.image_sets = self.dataset.image_sets
        self.preds = []
        self.gts = []
        self.evalfile = None

    def reset_results(self):
        self.preds = []
        self.gts = []
        return

    def eval_pr_curve(self, savedir):
        """
        :param :
        :param :
        :return: 
        """
        print("PR curve creating...")
        prelist = []
        reclist = []
        markthid = 0
        threadlist = [x/100.0 for x in range(5,100,5)]
        for idx,th in enumerate(threadlist):
            predict = np.array(self.preds)>th
            trueset = (np.array(self.preds)>th)==np.array(self.gts)
            falseset = 1-trueset
            TN = np.sum(predict&trueset,axis=0).astype(np.float32)
            TP = np.sum((1-predict)&trueset,axis=0).astype(np.float32)
            FN = np.sum(predict&falseset,axis=0).astype(np.float32)
            FP = np.sum((1-predict)&falseset,axis=0).astype(np.float32)
            precision = TP/(TP+FP+1e-7)
            recall = TP/(TP+FN+1e-7)
            score = 0.6*precision+0.4*recall
            print('thread:{},score:{}'.format(th,score))
            if TP.any()==0:
                print("warning!!! some of TP is ZERO at thread:{}...".format(th))
            prelist.append(precision)
            reclist.append(recall)
#             print(precision,recall)
            if th==0.5:
                markthid = idx
#                 import pdb
#                 pdb.set_trace()
        prearr = np.array(prelist)
        recarr = np.array(reclist)
        
        for idx,labelname in enumerate(labelmap):
            plt.figure(figsize=(8, 8))
            plt.title(labelname)
            plt.grid(True)
            plt.axis([0, 1, 0, 1])
            plt.ylabel('Precision  TP/(TP+FP)')
            plt.xlabel('Recall  TP/(TP+FN)')
            plt.xticks(np.arange(0, 1, 0.1))
            plt.yticks(np.arange(0, 1, 0.1))
            precisions = prearr[:,idx].tolist()
            recalls = recarr[:,idx].tolist()
            default_th_pr = [precisions[markthid],recalls[markthid]]
            
            plt.plot(recalls, precisions, label=labelname)
            plt.plot(recalls, precisions, 'o')
            plt.plot((default_th_pr[1]), (default_th_pr[0]), '*')
            plt.legend()

            nsave_figname = 'PR'+str(idx)+'_'+labelname + '.png'
            plt.savefig(os.path.join(savedir, nsave_figname))
            plt.close()
        
        truef = os.path.join(savedir, 'truedata.txt')
        falsef = os.path.join(savedir, 'falsedata.txt')
        
        TD = (np.array(self.preds)>0.5)==np.array(self.gts)
        FD = ~TD
        imgarr = np.array(self.dataset.ids)
        predarr = np.hstack(np.array(self.preds))
        
        truedata = imgarr[TD[:,0]].tolist()
        falsedata = imgarr[FD[:,0]].tolist()
        truepred = predarr[TD[:,0]].tolist()
        falsepred = predarr[FD[:,0]].tolist()
        print('creating true data file...')
        with open(truef,'w') as wf:
            for idx,img in enumerate(truedata):
                img = img.strip()
                pred = str(round(truepred[idx],3))
                print(img+' '+pred,file=wf)
        print('creating false data file...')
        with open(falsef,'w') as wf:
            for idx,img in enumerate(falsedata):
                img = img.strip()
                pred = str(round(falsepred[idx],3))
                print(img+' '+pred,file=wf)
        print('created all files')
        return

    def visfalse(self,savedir):
        falsef = os.path.join(savedir, 'falsedata.txt')
        falsedir = os.path.join(savedir, 'FALSE_IMG')
        fp_dir = os.path.join(falsedir, 'FP')
        fn_dir = os.path.join(falsedir, 'FN')
        if not os.path.exists(falsedir):
            print('creating false image view folders...')
            os.mkdir(falsedir)
            os.mkdir(fp_dir)
            os.mkdir(fn_dir)
        with open(falsef,'r') as rf:
            lines = rf.readlines()
        print('drawing false images...')
        for line in (lines):
            img,label,conf = line.strip().split(' ')
            basename = os.path.basename(img)
            IMG = Image.open(img)
            draw = ImageDraw.Draw(IMG)
            draw.text((0,0),conf,fill=(255,0,0))
            if label == '1':
                savename = os.path.join(fn_dir,basename)
                IMG.save(savename,'JPEG')
            elif label == '0':
                savename = os.path.join(fp_dir,basename)
                IMG.save(savename,'JPEG')
            else:
                print('ERROR LABEL， NOT 0 OR 1')
        return

    def post_proc(self, preds, targets):
        return NotImplementedError

    def evaluate_stats(self, classes=None, tb_writer=None):
        assert len(self.preds)==len(self.gts)
        gtap = sum((np.array(self.preds)>0.5)==np.array(self.gts))/len(self.gts)
        for idx,gt_name in enumerate(labelmap):
            print("gt class: {}; AP is: {}".format(gt_name,gtap[idx]))
        return np.mean(gtap)

    def validate(self, net, use_cuda=True, tb_writer=None):
        print('start evaluation')
#         self.reset_results()
        self.preds = []
        self.gts = []
        img_idx = 0
        _t = {'im_detect': Timer(), 'misc': Timer()}
        _t['misc'].tic()
        sigout = nn.Sigmoid()
        
        for batch_idx, (images, targets, extra) in enumerate(self.data_loader):
            if batch_idx % 25 == 0:
                print('processed image batch', batch_idx)
            if use_cuda:
                images = Variable(images.cuda(), volatile=True)
                extra = extra.cuda()
            else:
                images = Variable(images, volatile=True)

            _t['im_detect'].tic()
            preds = sigout(net(images))
            _t['im_detect'].toc(average=False)
            
            self.preds += preds.data.tolist()
            for gt in targets:
                self.gts.append(gt.numpy().tolist())
#             self.gts += targets
            
            # the format is now xmin, ymin, xmax, ymax, score, image, cls, (cocoid)
            if tb_writer is not None and tb_writer.cfg['show_test_image']:
                self.visualize_box(images, targets, h, w, det, img_idx, tb_writer)
#             img_idx = self.post_proc(preds,targets)

        _t['misc'].toc(average=False)
        # print(_t['im_detect'].total_time, _t['misc'].total_time)
        return self.evaluate_stats(None, tb_writer)
        
        
