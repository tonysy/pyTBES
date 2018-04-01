
from skimage.segmentation import mark_boundaries
from skimage.segmentation import find_boundaries
import copy
import numpy as np

class ChainCode(object):
    def __init__(self, image_superpixel):
        super(ChainCode, self).__init__()
        self.superpixel = image_superpixel
        self.boundary = find_boundaries(self.superpixel, mode='thick').astype(np.uint8)

        self.width = self.superpixel.shape[0]
        self.height = self.superpixel.shape[1]

        self.boundary_edge = find_boundaries(self.superpixel, mode='thick').astype(np.uint8)
        self.fill_egde()

    def chain_code(self, regin_index):
        pass

    def get_region_edge(self, region_id):
        """
        Args:
            region_id:
                reginon label in superpixels
        """
        # pixel locations  in this region
        region_pixels_loc = np.where(self.superpixel==region_id)
        region_boundary_value = self.boundary_edge[region_pixels_loc]
        region_boundary_loc = np.where(region_boundary_value==1)

        # import pdb ; pdb.set_trace()

    def fill_egde(self):
        """
        Change the edge value from 0 to 1
        """
        self.boundary_edge[0,:] = np.ones(self.height)
        self.boundary_edge[self.width-1,:] = np.ones(self.height)
        self.boundary_edge[:,0] = np.ones(self.width)
        self.boundary_edge[:,self.height-1] = np.ones(self.width)

      