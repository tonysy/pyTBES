from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import visdom 

from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from data.dataloader import Dataloader
import matplotlib.pyplot as plt 
# from lib.text_future import get_text_future
from lib.segmentor import Segmentor

bcd500_loader = Dataloader(data_dir='./dataset/BSR')
train_image = bcd500_loader.get_image(mode='train')

segmentor = Segmentor(train_image[0])
segmentor.get_texture_len(10,1)

import pdb ; pdb.set_trace()

segments_slic = slic(train_image[1], n_segments=100, compactness=10, sigma=1)

vis = visdom.Visdom(env='TBES_Visual_Results')
print('SLIC number of segments: {}'.format(len(np.unique(segments_slic))))
# import time
# s = time.time()

feature = get_text_future(train_image[1])
# print(time.time()-s)
plt.imshow(mark_boundaries(train_image[1], segments_slic))

if vis.win_exists('slic_superpixels_results'):
    vis.close(win='slic_superpixels_results')
    assert not vis.win_exists('slic_superpixels_results'), 'Closed window still exists'

vis.matplot(plt, win='slic_superpixels_results')


# temp = mark_boundaries(train_image[1], segments_slic)
# vis.image(temp.transpose(2,0,1))
