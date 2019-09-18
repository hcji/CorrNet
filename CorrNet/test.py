# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 10:41:30 2019

@author: hcji
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from CorrNet.main import CorrNet

data_l = np.load('Data/data_l.npy')
data_r = np.load('Data/data_r.npy')
label = np.load('Data/data_label.npy')

test_l = np.load('Data/test_v1.npy')
test_r = np.load('Data/test_v2.npy')
test_label = np.load('Data/test_l.npy')

corrnet = CorrNet(data_l, data_r)
corrnet.train()
rebu_l = corrnet.right_to_left(test_r)
rebu_r = corrnet.left_to_right(test_l)

def visualize(left, right):
    img_lef = left.reshape((28,14))
    img_rig = right.reshape((28,14))
    f, axarr = plt.subplots(1,2,sharey=False)
    axarr[0].imshow(img_lef)
    axarr[1].imshow(img_rig)

# from left to right
left = test_l[0]
right = rebu_r[0]
visualize(left, right)

# from right to left
left = rebu_l[0]
right = test_r[0]
visualize(left, right)

# latent correlation
latent1 = corrnet.left_to_latent(test_r)
latent2 = corrnet.right_to_latent(test_l)
pearsonr(latent1[0], latent2[0])
