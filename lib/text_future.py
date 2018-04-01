import numpy as np 
# try:
#     from .im2col.im2col_cython import col2im_cython, im2col_cython
#     from .im2col.im2col_cython import col2im_6d_cython
# except ImportError:
#     print('Run the following in root of project directory and try again')
#     print('cd ./lib/im2col/ && python setup.py build_ext --inplace && cd ../../')
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed

from lib.im2col import im2col
from sklearn.decomposition import PCA 
pca = PCA(n_components=8)
# @jit


class Segmentor(object):
    def __init__(self, image, superpixel=slic, pca_dim = 8):
        super(Segmentor, self).__init__()
        self.image_data = image
        self.image_super = superpixel(image, n_segments=200, \
                                        compactness=10, sigma=1)
        self.D = pca_dim
        self.kernel = 5 # size of window
        self.size = image.shape
        self.feature_dict = self.get_pixel_feature()

    def get_region_list(self):
        """Get pixel's region label
        For each group in superpixel map, get pixels' index in this group
        Args:

        Return:
            region_dict: dict

        """
        self.num_region = np.unique(self.image_super)
        self.super_flatten = self.image_super.reshape(-1)
        self.region_dict = {}
        for item in self.num_region:
            positions = np.argwhere(self.super_flatten==item)
            self.region_dict[item] = positions
    
        # self.region_dict
    def get_texture_len(self, region_ids, kernel):
        region_feature = self.feature_dict[kernel][region_ids].squeeze()
        region_mean = np.mean(region_feature, axis=0)
        region_cov = np.cov(region_feature)
        import pdb; pdb.set_trace()
        mean_term = self.D / 2 * np.log(2)*(1+ region_mean*region_mean.T)

    def get_pixel_feature(self):
        """
        Get Texture image feature for each image
        Use w-window to generate feature vector for each pixel
        """
        image_data = np.expand_dims(self.image_data, axis=-1)
        feature_dict = {}
        for i in range(1,self.kernel+1,2):
            feature_vector = im2col(image_data, HF=i,WF=i,pad=(i-1)/2,stride=1) # no padding(or add padding = 2)    
            feature_vector = feature_vector.T
            if i > 1:
                pca.fit(feature_vector)
                feature_vector = pca.transform(feature_vector)
            feature_dict[i] = feature_vector
        return feature_dict

