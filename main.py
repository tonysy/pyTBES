from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from data.dataloader import Dataloader


bcd500_loader = Dataloader(data_dir='./dataset/BSR')
train_image = bcd500_loader.get_image(mode='train')

segments_slic = slic(train_image[0], n_segments=50, compactness=10, sigma=1)
print('SLIC number of segments: {}'.format(len(np.unique(segments_slic))))