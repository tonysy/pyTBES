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
vis = visdom.Visdom(env='TBES_Visual_Results_NEW')

bcd500_loader = Dataloader(data_dir='./dataset/BSR')
train_image = bcd500_loader.get_image(mode='train')

segmentor = Segmentor(train_image[1])
# vis.image(mark_boundaries(segmentor.image_data,segmentor.image_super).transpose(2,0,1))

region_adjacency, _ = segmentor.get_region_adjacency_matrix()

# boundary_region_index = np.where(segmentor.image_super==1)
# boundary_coordinate = segmentor.chain_coder.get_region_edge_v2(boundary_region_index)
# vis.image(segmentor.chain_coder.region_boundary_mask*125)

for item in region_adjacency[1]:
    # boundary_region_index = np.where(segmentor.image_super==item)
    # boundary_coordinate = segmentor.chain_coder.get_region_edge_v2    (boundary_region_index)
    # vis.image(segmentor.chain_coder.region_boundary_mask*125)
    region_diff_len = segmentor.get_region_difference(1, item, kernel=7)
    print('Item:', item, region_diff_len)

import pdb ; pdb.set_trace()
boundary_coordinate = segmentor.chain_coder.get_region_edge(region_id=0)

# boundary_plot = segmentor.chain_coder.boundary_plot 
# for item in boundary_coordinate:
    # boundary_plot[item]=1

# boundary_plot *= 100


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
# plt.imshow(boundary_plot)

# plt.plot(boundary_coordinate[:,0],boundary_coordinate[:,1],'x')
# vis.matplot(plt,\
            # win='slic_superpixels_boundary_edge')
# boundary_plot = mark_boundaries(segmentor.image_data,segmentor.chain_coder.boundary_plot)
vis.image(segmentor.chain_coder.boundary_plot*100,\
            win='slic_superpixels_boundary_edge')



if vis.win_exists('slic_superpixels_region'):
    vis.close(win='slic_superpixels_region')
    assert not vis.win_exists('slic_superpixels_region'), 'Closed window still exists'
# boundary_edge = segmentor.chain_coder.boundary_edge
# boundary_edge *= 10
# plt.imshow(boundary_plot)

# plt.plot(boundary_coordinate[:,0],boundary_coordinate[:,1],'x')
# vis.matplot(plt,\
            # win='slic_superpixels_boundary_edge')
vis.image(segmentor.chain_coder.region_plot*100,\
            win='slic_superpixels_region')
# plt.imshow(mark_boundaries(train_image[1], segments_slic))

# boundary_path = segmentor.chain_coder.chain_code(boundary_coordinate)
import pdb ; pdb.set_trace()

# print(time.time()-s)


vis.matplot(plt, win='slic_superpixels_results')


# temp = mark_boundaries(train_image[1], segments_slic)
# vis.image(temp.transpose(2,0,1))
