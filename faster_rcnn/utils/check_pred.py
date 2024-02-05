import numpy as np

path = '/media/chs.gpu/DATA/hdd/chs.data/research-cancerPathology/aakash-rao-capstone-project/outputs/detectron/faster_rcnn_R_DC5_1x/results/results.npy'
results = np.load(path, allow_pickle=True)
print(results)