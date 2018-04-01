import numpy as np 
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from lib.im2col import im2col
from sklearn.decomposition import PCA 
from lib.chain_code import ChainCode

pca = PCA(n_components=8)

class Segmentor(object):
    def __init__(self, image, superpixel=slic, pca_dim = 8, epsilon=400):
        super(Segmentor, self).__init__()
        self.image_data = image
        self.image_super = superpixel(image, n_segments=100, \
                                        compactness=10, sigma=1)
        self.D = pca_dim
        self.distortion = epsilon
        self.kernel = 7 # size of window
        self.size = image.shape
        self.feature_dict = self.get_pixel_feature()
        self.region_dict = self.get_region_dict()
        # correponding to difference code
        # 0,1,2,3,4,5,6,7
        self.prob_chain_code =  [0.584800772633366,
                                 0.189661746246865,
                                 0.020453687819463,
                                 0.0,
                                 0.002104190215308,
                                 0.002890183173518,
                                 0.030969517355864,
                                 0.169119902555616]
        
        self.chain_coder = ChainCode(self.image_super)
        # import pdb; pdb.set_trace()


    def get_region_dict(self):
        """Get pixel's region label
        For each group in superpixel map, get pixels' index in this group
        Args:

        Return:
            region_dict: dict

        """
        self.num_region = np.unique(self.image_super)
        self.super_flatten = self.image_super.reshape(-1)
        region_dict = {}

        for item in self.num_region:
            positions = np.where(self.super_flatten==item)
            region_dict[item] = positions

        return region_dict
    
    # def get_edge_location(self):

    def get_texture_len(self, region_id, kernel):
        """
        Use equation(4) to calculate the coding length
        Attention: For easy to implement, we just use features of all pixel in a region, not chooese the nonoverlapping window.
        
        Args:
            region_id: 
                denotes for the index of the region of an image
            kernel:
                size of the window(e.g. 1, 3, 5 and 7)
        Return:
            length:
                coding length of one region under a window size
        """
        region_pixels = self.region_dict[region_id]
        N = region_pixels[0].shape[0] # number of pixel in a region
        region_feature = self.feature_dict[kernel][region_pixels]#.squeeze()
        import pdb ; pdb.set_trace()
        region_mean = np.mean(region_feature, axis=0)
        region_cov = np.cov(region_feature.transpose(1,0))
        mean_term = self.D / 2 * np.log2(1 + \
                    np.linalg.norm(region_mean)**2 \
                    / self.distortion**2)
        # first_term = (float(self.D) / 2  +  N/(2*kernel*kernel))*np.log2(np.linalg.det(1+float(self.D)*region_cov/(float(self.distortion)**2)))
        # TODO: Check why error if use np.linalg.det
        first_term = (float(self.D) / 2  +  N/(2*kernel*kernel))*np.sum(np.log2(1+ float(self.D)/(self.distortion**2)*np.linalg.svd(region_cov)[1]))

        length = first_term + mean_term
        assert length > 0

        # import pdb; pdb.set_trace()
        return length

    def get_boundary_len(self):
        pass

    def get_difference_chain_code(self):
        pass
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

