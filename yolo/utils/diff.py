import os
import numpy as np

l1 = os.listdir('/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/datasets/yolov5/fold_1/val/images')
l2 = os.listdir('/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/datasets/yolov5/test/images')

l1 = np.array(l1)
l2 = np.array(l2)

# finding common files in two lists
