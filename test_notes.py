import torch
import numpy as np
import argparse
import os
import sys
import time
import datetime
from ts2vec import TS2Vec
import tasks
import datautils
from utils import init_dl_program, name_with_datetime, pkl_save, data_dropout
import utils_dtw.data_augmentation as aug
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import math


UCR_minisets = ['ElectricDevices', 'ECGFiveDays', 'FaceAll', 'Crop', 'CricketX', 'GunPoint', 'ChlorineConcentration', 'InsectWingbeatSound','ItalyPowerDemand', 'Phoneme', 'StarLightCurves', 'CBF', 'SyntheticControl','Strawberry', 'Chinatown','MelbournePedestrian', 'ArrowHead' , 'PhalangesOutlinesCorrect' ,'AllGestureWiimoteZ']
''' 
UCR miniset description

각 Type에서 최소 하나씩은 선정하려고 노력했는데 length가 너무 길어서 학습 결과 너무 보기 오래 걸리는 경우에는 그냥 포기했음. 특히 ArrowHead / PhalangesOutlinesCorrect / AllGestureWiimoteZ 의 경우에는 Ts2Vec에서 임베딩 잘 되었다고 특히 강조했던 데이터셋이라서 포함 시켰음

1. ElectricDevices (Device)
- 8926 train data, 7711 test data
- 7 classes, length 96

2. ECGFiveDays (ECG)
- 861 train data, 23 test data
- 2 classes, length 136

3. FaceAll (Image)
- 1690 train data, 560 test data
- 14 classes, length 131

4. Crop (Image)
- 16800 train data, 7200 test data
- 24 classes, length 46

5. CricketX (Motion)
- 390 train data, 390 test data
- 12 classes, length 300

6. ChlorineConcentration (Sensor)
- 3840 train data, 467 test data
- 3 classes, length 166

7. InsectWingbeatSound (Sensor)
- 1980 train data, 220 test data
- 11 classes, length 256

8. ItalyPowerDemand (Sensor)
- 67 test data, 1029 train data
- 2 classes, length 24

9. Phoneme (Sensor)
- 214 test data, 1896 train data
- 39 classes, length 1024

10. StarLightCurves (Sensor)
- 1000 test data, 8236 train data
- 3	classes, length 1024

11. CBF (Simulated)
- 30 test data, 900 train data
- 3 classes, length 128

12. SyntheticControl (Simulated)
- 300 test data, 300 train data
- 6 classes, length	60

13. Strawberry (Spectro)
- 613 test data, 370 train data
- 2 classes, length 235

14. Chinatown (Traffic)
- 20 test data, 343 train data
- 2 classes, length 24

15. MelbournePedestrian (Traffic)
- 1194 test data, 2439 train data
- 10 classes, length 24

16. ArrowHead (Image)
- 36 test data, 175 train data
- 3 classes, length 251

17. PhalangesOutlinesCorrect (Image)
- 1800 test data, 858 train data
- 2 classes, length 80

18. AllGestureWiimoteX (Sensor)
- 300 test data, 700 train data
- 10 classes, length Vary (but interpolated all data to match to the longest)
'''

DTW_AUGS_DICT = {
    "rgw_sdtw" : aug.random_guided_warp_shape,
    "rgw_dtw" :aug.random_guided_warp,
    "dgw_sdtw" : aug.discriminative_guided_warp_shape,
    "dgw_dtw" : aug.discriminative_guided_warp,
}
# seeds 0, 1, 42, 1557, 1601, 777, 2453, 1238, 1360, 1843, 1917, 1004, 466, 497, 1317

print('Loading data... ', end='')
original_root = Path('/data1/finance/UCR')
augment_root = Path('/data1/finance/UCR_augmented')

seed = 0
aug_name = list(DTW_AUGS_DICT.keys())[2]

# TODO: set path
for dataset in UCR_minisets:
    aug_joined_root = os.path.join(augment_root, aug_name, str(seed))
    ori_joined_root = os.path.join(original_root)

    ori_train, ori_train_labels, ori_test, ori_test_labels = datautils.load_UCR(dataset, original_root)
    aug_train, aug_labels = datautils.load_UCR_aug(dataset, aug_joined_root)

    print(dataset, aug_name, np.unique(ori_train == aug_train, return_counts=True))
    assert (np.unique(aug_labels == ori_train_labels) == [True]) and (ori_train.shape[0] == aug_train.shape[0])


    

