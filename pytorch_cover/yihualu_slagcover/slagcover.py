import os
import sys
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
from torch.autograd import Variable

from PIL import Image

dataf = sys.argv[1]

nowdir = os.getcwd()

covermodel = os.path.join(nowdir,'modelsave/slagcover_20000_99.pth')
coverth = 0.6

clstransform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

datalist = []
with open(dataf,'r') as rf:
    datalist = rf.readlines()

covernet = torch.load(covermodel)['state_dict']
covernet = torch.nn.DataParallel(covernet)
covernet = covernet.cuda()
covernet.eval()

sigout = nn.Sigmoid()

savelist = []
for item in datalist:
    imgpath,label = item.strip().split(' ')
    baseimg = os.path.basename(imgpath)
    carimg = Image.open(imgpath)
    
    carimg = clstransform(carimg)
    carimg = carimg.unsqueeze(0)
    carimg = Variable(carimg.cuda(), volatile=True)
    conf = sigout(covernet(carimg)).cpu().data.numpy()[0][0]
#     predstr = "{}: {} label:{}".format(baseimg,conf>coverth,label)
    predstr = "{}: {}".format(baseimg,conf>coverth)
#     print(predstr)
    savelist.append(predstr)
    
print("creat data file")
with open('slagcover_pred.txt','w') as wf:
    for item in savelist:
        print(item,file=wf)