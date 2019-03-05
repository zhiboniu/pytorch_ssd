import os
import shutil
from tqdm import tqdm

dataf = 'covertest_new.txt'
savedir = 'slagcover_test100'

if not os.path.exists(savedir):
    os.mkdir(savedir)
    
with open(dataf,'r') as rf:
    lines = rf.readlines()
    
for line in tqdm(lines):
    imgpath,_ = line.split(' ')
    baseimg = os.path.basename(imgpath)
    newpath = os.path.join(savedir,baseimg)
    shutil.copy(imgpath,newpath)