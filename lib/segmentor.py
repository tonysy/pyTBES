import numpy as np 
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from lib.im2col import im2col
from sklearn.decomposition import PCA 
import skimage.morphology as sm
from lib.chain_code import ChainCode

pca = PCA(n_components=8)

class Segmentor(object):
    def __init__(self, image, superpixel=slic, pca_dim = 8, epsilon=800):
        super(Segmentor, self).__init__()
        self.image_data = image
        self.image_super = superpixel(image, n_segments=100, \
                                        compactness=10, sigma=1)
        self.num_region = np.unique(self.image_super)
        
        self.init_unique = len(np.unique(self.image_super))

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
                                 0.0+1e-13,
                                 0.002104190215308,
                                 0.002890183173518,
                                 0.030969517355864,
                                 0.169119902555616]
        # self.prob_chain_code =  [0.020453687819463,
        #                          0.189661746246865,
        #                          0.584800772633366,
        #                          0.169119902555616,
        #                          0.030969517355864,
        #                          0.002890183173518,
        #                          0.002104190215308,
        #                          0.0+1e-13]

        self.chain_coder = ChainCode(self.image_super)
        # import pdb; pdb.set_trace()


    def get_region_dict(self):
        """Get pixel's region label
        For each group in superpixel map, get pixels' index in this group
        Args:

        Return:
            region_dict: dict

        """
        self.super_flatten = self.image_super.reshape(-1)
        region_dict = {}

        for item in self.num_region:
            positions = np.where(self.super_flatten==item)
            region_dict[item] = positions

        return region_dict
    
    # def get_edge_location(self):

    def get_texture_len(self, region_pixels_index, kernel):
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
        # import pdb ; pdb.set_trace()
        N = region_pixels_index[0].shape[0] # number of pixel in a region
        region_feature = self.feature_dict[kernel][region_pixels_index]#.squeeze()
        # import pdb ; pdb.set_trace()
        region_mean = np.mean(region_feature, axis=0)
        region_cov = np.cov(region_feature.transpose(1,0))
        mean_term = self.D / 2 * np.log2(1 + \
                    np.linalg.norm(region_mean)**2 \
                    / self.distortion**2)
        # first_term = (float(self.D) / 2  +  N/(2*kernel*kernel))*np.log2(np.linalg.det(1+float(self.D)*region_cov/(float(self.distortion)**2)))
        # TODO: Check why error if use np.linalg.det
        first_term = (float(self.D) / 2  +  N/(2*kernel*kernel))*np.sum(np.log2(1+ float(self.D)/(self.distortion**2)*np.linalg.svd(region_cov)[1]))

        texture_length = first_term + mean_term
        assert texture_length > 0

        # import pdb; pdb.set_trace()
        return texture_length

    def get_boundary_len(self, boundary_region_index):
        
        diff_chain_code = self.get_diff_chain_code(boundary_region_index)
        key_counts = [diff_chain_code.count(i) for i in range(8)]
        weight_list = np.log2(np.array(self.prob_chain_code))

        boundary_length = -np.sum(np.array(key_counts).astype(np.float)*weight_list)
        
        # import pdb ; pdb.set_trace()
        return boundary_length

    def get_diff_chain_code(self, boundary_region_index):
        
        boundary_coordinate = self.chain_coder.get_region_edge_v2(boundary_region_index)

        freeman_chain_code = self.chain_coder.get_chain_code(boundary_coordinate)
        
        diff_chain_code = self.chain_coder.get_diff_chain_code(freeman_chain_code)
        print('Boundary_coordinate Size:', boundary_coordinate.shape[0])
        print('Chain Code Size:', len(diff_chain_code))
        return diff_chain_code

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

    def get_total_length_single(self,texture_region_index,\
                                boundary_region_index,kernel):
        single_total_length = 0
        for regin_id in self.num_region.tolist():
            # print(regin_id)
            # region_pixels_index = self.region_dict[region_id]
            texture_length = self.get_texture_len(texture_region_index,
                                                kernel=kernel)
            boundary_length = self.get_boundary_len(boundary_region_index)
            single_total_length += (texture_length + 0.5*boundary_length)

        return single_total_length

    def get_region_adjacency_matrix(self):
        region_adjacency_matrix = np.zeros((self.init_unique,\
                                            self.init_unique))
        region_adjacency_dict = {}

        for regin_id in self.num_region.tolist():
            region_original = np.zeros((self.chain_coder.height,\
                                      self.chain_coder.width))
            region_original[np.where(self.image_super==regin_id)] = 1

            region_dilation = sm.dilation(region_original,sm.square(3))
            overlapping_edge = np.logical_xor(region_original, region_dilation)
            
            adjacency_mask = self.image_super[overlapping_edge]
            adjacency_region_list = np.unique(adjacency_mask)
            
            indexs = (np.array([regin_id]*adjacency_region_list.shape[0]),\
                            adjacency_region_list)
            region_adjacency_matrix[indexs] = 1

            region_adjacency_dict[regin_id] = adjacency_region_list

        self.region_adjacency_dict = region_adjacency_dict 
        self.region_adjacency_matrix = region_adjacency_matrix
        return region_adjacency_dict, region_adjacency_matrix

    def get_region_difference(self, region_id_a, region_id_b,kernel):
        texture_region_index_a, \
        texture_region_index_b, \
        merge_texture_region = self.texture_region_merge(region_id_a, region_id_b)


        texture_len_a = self.get_texture_len(texture_region_index_a,kernel)
        texture_len_b = self.get_texture_len(texture_region_index_b,kernel)     
        merge_texture_len = self.get_texture_len(merge_texture_region,\
                                                kernel=kernel)


        # boundary_region_index_a, \
        # boundary_region_index_b, \
        # merge_boundary_region = self.boundary_region_merge(region_id_a, region_id_b)

        # boundary_len_a = self.get_boundary_len(boundary_region_index_a)
        # boundary_len_b = self.get_boundary_len(boundary_region_index_b)
        # merge_boundary_len = self.get_boundary_len(merge_boundary_region)

        region_diff_len = texture_len_a + texture_len_b \
                            - merge_texture_len \
                            # + 0.5*(boundary_len_a + \
                            # boundary_len_b - merge_boundary_len)

        return region_diff_len

    def boundary_region_merge(self, region_id_a, region_id_b):
        boundary_region_index_a = np.where(self.image_super==region_id_a)
        boundary_region_index_b = np.where(self.image_super==region_id_b)
    
        merge_boundary_region = (np.hstack((boundary_region_index_a[0],
                                        boundary_region_index_b[0])),
                               np.hstack((boundary_region_index_a[1],
                                        boundary_region_index_b[1])),)
        # import pdb ; pdb.set_trace()
        return boundary_region_index_a,\
               boundary_region_index_b, \
               merge_boundary_region
    def texture_region_merge(self, region_id_a, region_id_b):
        texture_region_index_a = self.region_dict[region_id_a]
        texture_region_index_b = self.region_dict[region_id_b]

        merge_texture_region = (np.hstack((texture_region_index_a[0],texture_region_index_b[0])),)
        return texture_region_index_a,\
               texture_region_index_b, \
               merge_texture_region

    def optimize_segmentation(self):
        kernel = self.kernel
        while True:
            adjacency_region_list = np.argwhere(self.region_adjacency_matrix==1)
            

            item_0 = adjacency_region_list[0]
            len_max_item = item_0
            region_diff_len_max = self.get_region_difference(item_0[0], item_0[1], kernel)

            # get maximum len item
            for item in adjacency_region_list:
                region_diff_len = self.get_region_difference(item[0], item[1], kernel)
                if region_diff_len > region_diff_len_max:
                    print('---New Largest Length:{}--------'.format(region_diff_len))
                    len_max_item = item
                    region_diff_len_max = region_diff_len

            if region_diff_len_max > 0:
                print('Prepare to Merge')
                # update superpixel
                self.merge_region(len_max_item[0],len_max_item[1])
                
            elif region_diff_len_max <=0 and kernel != 1:

                kernel = kernel - 2
            else: #if kernel == 1 && region_diff_len_max <= 0:
                break
        print('Merge Complete!')
        # import pdb ; pdb.set_trace()
                
    def merge_region(self, region_a, region_b):
        num_unique = len(np.unique(self.image_super))
        index_region_b = np.where(self.image_super==region_b)
        self.image_super[index_region_b] = region_a
        assert num_unique == len(np.unique(self.image_super))+1
        
        # Update region dict
        self.region_dict = self.get_region_dict()
        self.num_region = np.unique(self.image_super)
        self.get_region_adjacency_matrix()