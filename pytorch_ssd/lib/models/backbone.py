# -*- coding:utf-8 -*-
import torch
import torch.nn as nn

layer_config = {
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'C',
              512, 512, 512],
    'vgg16_self': [64, 64, 'S', 128, 128, 128, 'S', 256, 256, 256, 256, 'S', 512, 512, 512, 512, 'S',
              512, 512, 512, 512, 'C', 512, 512],
    'vgg16_lite': [64, 64, 'M', 128, 128, 'M', 256, 256, 'C', 512, 512, 'C', 512, 512, 'C', 512, 512],
    'mobilenetv1': [8, 8, 'S', 16, 16, 16, 'S', 32, 32, 32, 32, 32, 32, 32, 32, 'S', 64],
    'mobilenetv2': [8, 'S', 16, 16, 'S', 32, 32, 32, 32, 32, 32, 'S', 64],
}
L2Norm_config = {
    'vgg16': (13,512),
    'mobilenetv1': (3+12*6,64),
    'mobilenetv2': (3+8*1,64),
}
Source_config = {
    'vgg16': (-4,-1),
    'mobilenetv1': (-2,-1),
    'mobilenetv2': (-2,-1),
}

basestr = 'vgg16'

L2Norm_cfg = L2Norm_config[basestr]
Source_cfg = Source_config[basestr]

"""
layer_idx: start from 1, in code need to -1
batch_norm_false: 2 4 5(M) 7 9 10(M) 12 14 16 17(C) 19 21 23 24(M) 26 28 30
batch_norm_true:  3 6 7(M) 10 13 14(M) 17 20 23 24(C) 27 30 33 34(M) 37 40 43
"""

def backbone():
    if basestr=='vgg16':
        return vgg_cls()
    elif basestr=='mobilenetv1':
        return mobilenetv1(mult=2)
    elif basestr=='mobilenetv2':
        return mobilenetv2(mult=2)
    else:
        print("False! no such basemodel!")
    return

def vgg_cls(i=3, batch_norm=False):
    """This function is derived from torchvision VGG make_layers()
    https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py"""
    layer_cfg = layer_config['vgg16_lite']
    layers = []
    in_channels = i
    for idx,v in enumerate(layer_cfg):
        if in_channels == 'S':
            in_channels = v
            continue
        elif v == 'S':
            conv2d = nn.Conv2d(in_channels, layer_cfg[idx+1], kernel_size=3, stride=2, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
        elif v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]  # ceil: integer >= x
        else:
#             print('icoc:',in_channels,v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    pool5 = nn.MaxPool2d(kernel_size=7, stride=1, padding=0)
#     conv6 = nn.Linear(512, 1024, kernel_size=3, padding=6, dilation=6)
#     conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
#     layers += [pool5, conv6,
#                nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    layers += [pool5]
    return layers

class vgg16_cls(nn.Module):
    def __init__(self, layers, num_classes, phase):
        super(vgg16_cls, self).__init__()
        self.features = nn.Sequential(*layers)
        if phase=='train':
            self.classifier = nn.Sequential( #分类器结构
                #fc6
        #             nn.Linear(512, 1000),
        #             nn.ReLU(),
        #             nn.Dropout(),

                #fc7
        #             nn.Linear(4096, 1000),
        #             nn.ReLU(),
        #             nn.Dropout(),

                #fc8
        #             nn.Linear(1000, num_classes)
                nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0)
            )
        elif phase=='eval':
            self.classifier = nn.Sequential( #分类器结构
                #fc6
        #             nn.Linear(512, 1000),
        #             nn.ReLU(),
        #             nn.Dropout(),

                #fc7
        #             nn.Linear(4096, 1000),
        #             nn.ReLU(),
        #             nn.Dropout(),

                #fc8
        #             nn.Linear(1000, num_classes)
                nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid()
            )
        else:
            raise "no such phase..."
            
    def forward(self, x):
        x = self.features(x)
#         print('before x.size:{}'.format(x.shape))
#         x = x.view(x.size(0), -1)
        x = self.classifier(x)
#         print('after 1 x.size:{}'.format(x.shape))
        x = torch.squeeze(x,2)
#         print('after 2 x.size:{}'.format(x.shape))
        x = torch.squeeze(x,2)
#         print('after x.size:{}'.format(x.shape))
        return x
    

def vgg(i=3, batch_norm=False):
    """This function is derived from torchvision VGG make_layers()
    https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py"""
    layer_cfg = layer_config['vgg16']
    layers = []
    in_channels = i
    for v in layer_cfg:
        if in_channels == 'S':
            in_channels = v
            continue
        elif v == 'S':
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, stride=2, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
        elif v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]  # ceil: integer >= x
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    pool5 = nn.MaxPool2d(kernel_size=5, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]

    return layers

class mobilenetv1_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, batch_norm=True):
        super(mobilenetv1_block, self).__init__()
        self.out_channels = out_channels
        self.block_layer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels))
         
    def forward(self, x):
        block = self.block_layer(x)
        return block
            

# def mobilenetv1_block(in_channels, out_channels, stride, batch_norm=True):
#     block_layers = []
        
#     dwconv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
#     if batch_norm:
#         block_layers += [dwconv, nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True)]
#     else:
#         block_layers += [dwconv, nn.ReLU(inplace=True)]
#     pwconv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
#     if batch_norm:
#         block_layers += [pwconv, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
#     else:
#         block_layers += [pwconv, nn.ReLU(inplace=True)]
#     return block_layers

def mobilenetv1(mult=1, in_c=3, batch_norm=True):
    """This function is derived from torchvision VGG make_layers()
    https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py"""
    layer_cfg = layer_config['mobilenetv1']
    layer_cfg = [x*mult if type(x)==int else x for x in layer_cfg]
    print("mobilenetv1 layer_cfg",layer_cfg)
    layers = []
    
    in_channels = in_c
    conv1_out = layer_cfg[0]
    conv1 = nn.Conv2d(in_channels, conv1_out, kernel_size=3, stride=2, padding=1)
    if batch_norm:
        layers += [conv1, nn.BatchNorm2d(conv1_out), nn.ReLU(inplace=True)]
    else:
        layers += [conv1, nn.ReLU(inplace=True)]
    in_channels = conv1_out
        
    for index,layer_cfg_item in enumerate(layer_cfg[1:],1):
        if in_channels == 'S':
            in_channels = layer_cfg_item
            continue
        if layer_cfg_item == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif layer_cfg_item == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]  # ceil: integer >= x
        elif layer_cfg_item == 'S':
            layers.append(mobilenetv1_block(in_channels, layer_cfg[index+1], stride=2, batch_norm=True))
            in_channels = layer_cfg_item
        else:
            layers.append(mobilenetv1_block(in_channels, layer_cfg_item, stride=1, batch_norm=True))
            in_channels = layer_cfg_item

    return layers

class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, kernel_size=3, expansion_factor=3):
        super(InvertedResidualBlock, self).__init__()

        if stride != 1 and stride != 2:
            raise ValueError("Stride should be 1 or 2")

        self.out_channels = out_channels

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * expansion_factor, 1),
            nn.BatchNorm2d(in_channels * expansion_factor),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels * expansion_factor, in_channels * expansion_factor,
                      kernel_size, stride, 1,
                      groups=in_channels * expansion_factor),
            nn.BatchNorm2d(in_channels * expansion_factor),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels * expansion_factor, out_channels, 1),
            nn.BatchNorm2d(out_channels))

        self.is_residual = True if stride == 1 else False # 当该单元的stide = 1 时采用skip connection
        self.is_conv_res = False if in_channels == out_channels else True # 匹配输入 输出通道的一致性

        # Assumption based on previous ResNet papers: If the number of filters doesn't match,
        # there should be a conv1x1 operation.
        if stride == 1 and self.is_conv_res:
            self.conv_res = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1),
                                          nn.BatchNorm2d(out_channels))

    def forward(self, x):
        block = self.block(x)
        if self.is_residual:
            if self.is_conv_res:
                return self.conv_res(x) + block
            return x + block
        return block

def mobilenetv2(mult=1, in_c=3, batch_norm=True):
    layer_cfg = layer_config['mobilenetv2']
    layer_cfg = [x*mult if type(x)==int else x for x in layer_cfg]
    print("mobilenetv2 layer_cfg",layer_cfg)
    layers = []
    
    in_channels = in_c
    conv1_out = layer_cfg[0]
    conv1 = nn.Conv2d(in_channels, conv1_out, kernel_size=3, stride=2, padding=1)
    if batch_norm:
        layers += [conv1, nn.BatchNorm2d(conv1_out), nn.ReLU(inplace=True)]
    else:
        layers += [conv1, nn.ReLU(inplace=True)]
    in_channels = conv1_out
    
    for index,layer_cfg_item in enumerate(layer_cfg[1:],1):
        if in_channels == 'S':
            in_channels = layer_cfg_item
            continue
        if layer_cfg_item == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif layer_cfg_item == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]  # ceil: integer >= x
        elif layer_cfg_item == 'S':
            layers.append(InvertedResidualBlock(in_channels, layer_cfg[index+1], 2))
            in_channels = layer_cfg_item
        else:
            layers.append(InvertedResidualBlock(in_channels, layer_cfg_item, 1))
            in_channels = layer_cfg_item

    return layers