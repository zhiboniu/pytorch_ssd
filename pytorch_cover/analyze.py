import argparse

import pandas as pd
from matplotlib import patches
import matplotlib.pyplot as plt
from PIL import Image

from lib.datasets import dataset_factory
from lib.utils.config import cfg
from lib.utils.utils import setup_folder
from lib.datasets.voc0712 import VOC_CLASSES


class DetectVis(object):
    def __init__(self, img_list, show_axis=True):
        self.img_idx = 0
        self.img_list = img_list
        self.vis_funcs = []
        self.show_axis = show_axis

    @staticmethod
    def draw(ax, df, score=False, **kwargs):
        for idx, row in df.iterrows():
            edgecolor = kwargs['color'][int(row['class'])] if 'color' in kwargs else 'b'
            linestyle = kwargs['linestyle'] if 'linestyle' in kwargs else '-'
            rect = patches.Rectangle((row['xmin'], row['ymin']), (row['xmax'] - row['xmin']),
                                    (row['ymax'] - row['ymin']), linewidth=1, edgecolor=edgecolor,
                                     facecolor='none', linestyle=linestyle)
            ax.add_patch(rect)
            if score:
                ax.text(row['xmin'], row['ymax'], row['name'] + ' ' + '{:.2f}'.format(row['score']),
                        color='yellow', fontsize='large')

    @staticmethod
    def colored_box(multiply, score, print_df=True, **kwargs):
        def colored_box(ax, df, path, img):
            """
            show picture given an dataframe contatains annotaion
            :param ax: matplotlib axis
            :param df: pandas dataframe contains annotation
            """
            w, h = img.size
            df = df[df.path == path]
            if multiply:
                df.xmin *= w
                df.xmax *= w
                df.ymin *= h
                df.ymax *= h
            DetectVis.draw(ax, df, score, **kwargs)
        return colored_box

    def show_detections(self, vis_pair):
        """
        show image with detection annotation. press any key to see next annotated image
        :param vis_pair: list of pair of pandas dataframe and vis function
        """
        fig, ax = plt.subplots(1)

        def onclick(event):
            if self.img_idx == len(self.img_list):
                plt.close()
                return
            plt.cla()
            path = self.img_list[self.img_idx]
            plt.title(path)
            img = Image.open(path)
            ax.imshow(img, cmap='gray')
            # draw boxes
            for df, func in vis_pair:
                func(ax, df, path, img)
            if not self.show_axis:
                plt.axis('off')
            fig.canvas.draw()
            self.img_idx += 1

        cid = fig.canvas.mpl_connect('key_press_event', onclick)
        mng = plt.get_current_fig_manager()
        # mng.window.showMaximized()
        plt.show()


def read_gt():
    loader = dataset_factory(phase='eval', cfg=cfg)
    dataset = loader.dataset
    gts = []
    bs = cfg.DATASET.EVAL_BATCH_SIZE
    for batch_idx, (images, targets, etc) in enumerate(loader):
        start = batch_idx*bs
        end = min(batch_idx*bs + bs, len(dataset))
        img_paths = [dataset._imgpath % dataset.ids[idx] for idx in range(start, end)]
        for p, t in zip(img_paths, targets):
            df = pd.DataFrame(t.numpy(), columns=['xmin', 'ymin', 'xmax', 'ymax', 'class'])
            df_head = pd.DataFrame([[p]], columns=['path'], index=range(len(t)))
            df = pd.concat([df_head, df], axis=1)
            gts.append(df)
    gt = pd.concat(gts, axis=0, ignore_index=True)
    with pd.HDFStore('./cache/voc.hdf') as hdf:
        hdf.put('gt', gt, format='table', data_columns=True)
    print(gt)


def analyze_gt(tb_writer):
    # read_gt()
    gt = pd.read_hdf('./cache/voc.hdf')
    vis = DetectVis(img_list=gt.path.unique())
    # color = ['w', 'b', 'r', 'y', 'c', 'k']
    vis_pair = [(gt, DetectVis.colored_box(multiply=True, score=False, linestyle='-'))]
    vis.show_detections(vis_pair)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Single Shot MultiBox Detector Training With Pytorch')
    parser.add_argument('--cfg_name', default='ssd_analyze_voc',
                        help='base name of config file')
    parser.add_argument('--job_group', default='base', type=str,
                        help='Directory for saving checkpoint models')
    parser.add_argument('--cuda', default=False, type=bool,
                        help='Use CUDA to train model')
    parser.add_argument('--tensorboard', default=True, type=bool,
                        help='Use tensorboard')
    parser.add_argument('--devices', default='0,1,2,3,4', type=str,
                        help='GPU to use for net forward')
    parser.add_argument('--net_gpus', default=[0,1,2,3], type=list,
                        help='GPU to use for net forward')
    parser.add_argument('--loss_gpu', default=4, type=list,
                        help='GPU to use for loss calculation')
    args = parser.parse_args()
    tb_writer, _, _, log_dir = setup_folder(args, cfg)
    analyze_gt(tb_writer)

