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
segmentor.get_texture_len(region_id=0, kernel=1)

segmentor.chain_coder.get_region_edge(region_id=0)
import pdb ; pdb.set_trace()

vis = visdom.Visdom(env='TBES_Visual_Results')
# print('SLIC number of segments: {}'.format(len(np.unique(segments_slic))))

# plot superpixel
if vis.win_exists('slic_superpixels_results'):
    vis.close(win='slic_superpixels_results')
    assert not vis.win_exists('slic_superpixels_results'), 'Closed window still exists'
# import pdb ; pdb.set_trace()

vis.image(mark_boundaries(segmentor.image_data, \
                        segmentor.image_super).transpose(2,0,1), \
                        win='slic_superpixels_results')

boundary = segmentor.chain_coder.boundary
boundary *= 200
if vis.win_exists('slic_superpixels_boundary'):
    vis.close(win='slic_superpixels_boundary')
    assert not vis.win_exists('slic_superpixels_boundary'), 'Closed window still exists'
vis.image(boundary,\
            win='slic_superpixels_boundary')

if vis.win_exists('slic_superpixels_boundary_edge'):
    vis.close(win='slic_superpixels_boundary_edge')
    assert not vis.win_exists('slic_superpixels_boundary_edge'), 'Closed window still exists'
# boundary_edge = segmentor.chain_coder.boundary_edge
# boundary_edge *= 10
# plt.imshow(boundary_edge)

# vis.matplot(plt,\
#             win='slic_superpixels_boundary_edge')
# vis.image(boundary_edge,\
            # win='slic_superpixels_boundary_edge')

# plt.imshow(mark_boundaries(train_image[1], segments_slic))

import pdb ; pdb.set_trace()

# print(time.time()-s)


vis.matplot(plt, win='slic_superpixels_results')


# temp = mark_boundaries(train_image[1], segments_slic)
# vis.image(temp.transpose(2,0,1))
