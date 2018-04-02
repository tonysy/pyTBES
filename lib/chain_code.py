
from skimage.segmentation import mark_boundaries
from skimage.segmentation import find_boundaries
import copy
import numpy as np

class ChainCode(object):
    def __init__(self, image_superpixel):
        super(ChainCode, self).__init__()
        self.superpixel = image_superpixel
        self.boundary = find_boundaries(self.superpixel, mode='thick').astype(np.uint8)

        self.width = self.superpixel.shape[1]
        self.height = self.superpixel.shape[0]

        self.boundary_filledge = find_boundaries(self.superpixel, mode='thick').astype(np.uint8)
        self.fill_egde()

    
    def get_chain_longest_path(self, start_point, boundary_coordinate):
        """
        Generate chain code and point path.
        Use boundary_coordinate to generate its chain code

        We use code formulation as follows:
            7   0   1
             \  |  /
              \ | /             
          6 -------- 2
              / | \
             /  |  \
            5   4   3
    
        """
        current_point = start_point

        while np.sum(self.flag_list) < boundary_coordinate.shape[0]:
            neighbors = self.get_neighbor_coordinate_dis1(current_point)
            # list_direction = sorted(range(8), reverse=True)
            # for i in range(7,0,1):
            list_direction = [2,1,0,7,6,5,4,3]
            for i in list_direction:
                count = 0
                if neighbors[i].tolist() in boundary_coordinate.tolist():
                    neighbor_idx = boundary_coordinate.tolist().index(neighbors[i].tolist())
                    # check whether this points is used or not
                    if self.flag_list[neighbor_idx] == 0:            
                        # import pdb ; pdb.set_trace()
                        self.flag_list[neighbor_idx] = 1
                        self.freeman_chain_code_list.append(i)
                        self.chain_points.append(neighbors[i])
                        current_point = neighbors[i]
                        count += 1
                        break
                    else:
                        pass
            if count == 0:
                break

        stuck_point = current_point
        return stuck_point

    def get_chain_code(self,boundary_coordinate):
        """
        Generate freeman chain code,
        Skip the stuck point
        """
        self.flag_list = np.zeros(boundary_coordinate.shape[0])

        start_point = boundary_coordinate[0]
        new_start_point = copy.deepcopy(start_point)
        self.freeman_chain_code_list = []
        self.chain_points = []

        while True:
            stuck_point = self.get_chain_longest_path(start_point, \
                                            boundary_coordinate)

            # exit
            if np.sum(self.flag_list) == boundary_coordinate.shape[0]:
                print('Wonderfule! No Stuck point!')
                break
            
            # stucked, need to generate new start point
            neighbors = self.get_neighbor_coordinate_dis2(stuck_point)
            for i in range(16):
                if neighbors[i].tolist() in boundary_coordinate.tolist():
                    neighbor_idx = boundary_coordinate.tolist().index(neighbors[i].tolist())
                    if self.flag_list[neighbor_idx] == 0:
                        new_start_point = neighbors[i]
                        break
            # import pdb ; pdb.set_trace()
            print('Total Length:',np.sum(self.flag_list))          

            if new_start_point.tolist() == start_point.tolist():
                break

            # Update for new start point
            start_point = new_start_point
        return self.freeman_chain_code_list

    def get_diff_chain_code(self, free_chain_code):
        """
        Generate diff chain code from freeman chain code
        """
        diff_chain_code = []
        start_code =  free_chain_code[0]
        diff_chain_code = [np.mod(free_chain_code[i]-free_chain_code[i-1],8) for i in range(1,len(free_chain_code))]

        diff_chain_code.insert(0, start_code)

        return diff_chain_code

    
    def get_region_edge(self, region_id):
        """
        Args:
            region_id:
                reginon label in superpixels
        """
                
        # step-1: get all pixels index of a region from superpixel, 
        # for generate boundary mask
        region_pixels_index = np.where(self.superpixel==region_id)
        
        # step-2: get region boundary mask
        region_boundary_mask = self.boundary_filledge[region_pixels_index]
        # region_boundary_mask = self.boundary[region_pixels_index]
        
        # step-3: get all pixels index of a region for mapping
        # region_pixels_loc = np.argwhere(self.superpixel==region_id)        
        num_pixels = region_pixels_index[0].shape[0] # number of pixels in a region
        region_pixels_loc_tuple = [(region_pixels_index[0][i],region_pixels_index[1][i]) for i in range(num_pixels)]

        # step-4: generate dict for mapping boundary mask point index and its coordinate
        mapping_dict = dict(zip(range(num_pixels), region_pixels_loc_tuple))
        
        # step-5: generate boundary point index from boundary mask
        region_boundary_point_index = np.where(region_boundary_mask==1)

        # step-6: use region boundary_point_index to look up its coordinate
        # np.where return a tuple, fetch the np.ndarray with index [0]

        boundary_coordinate = np.array([mapping_dict[key] for key in region_boundary_point_index[0].tolist()])
        

        # For plot
        # self.boundary_plot = np.zeros((self.height,self.width))
        # self.boundary_plot[(boundary_coordinate[:,0],boundary_coordinate[:,1])] = 1

        self.region_plot = np.zeros((self.height,self.width))
        self.region_plot[np.where(self.superpixel==region_id)]=1

        return boundary_coordinate

    def get_region_edge_v2(self, boundary_region_index):
        """
        Args:
            region_id:
                region label in superpixels
        """
        region_initial = np.zeros((self.height,self.width))
        region_initial[boundary_region_index]=1
        region_boundary_mask = find_boundaries(self.superpixel, mode='inner',background=0).astype(np.uint8)

        # step-3: get all pixels index of a region for mapping
        # region_pixels_loc = np.argwhere(self.superpixel==region_id)        
        num_pixels = boundary_region_index[0].shape[0] # number of pixels in a region
        region_pixels_loc_tuple = [(boundary_region_index[0][i],boundary_region_index[1][i]) for i in range(num_pixels)]

        # step-4: generate dict for mapping boundary mask point index and its coordinate
        mapping_dict = dict(zip(range(num_pixels), region_pixels_loc_tuple))
        
        # step-5: generate boundary point index from boundary mask
        region_boundary_point_index = np.where(region_boundary_mask==1)

        # step-6: use region boundary_point_index to look up its coordinate
        # np.where return a tuple, fetch the np.ndarray with index [0]

        boundary_coordinate = np.array([mapping_dict[key] for key in region_boundary_point_index[0].tolist()])

        return boundary_coordinate

    def fill_egde(self):
        """
        Change the edge value from 0 to 1
        """
        self.boundary_filledge[0,:] = np.ones(self.width)
        self.boundary_filledge[self.height-1,:] = np.ones(self.width)
        self.boundary_filledge[:,0] = np.ones(self.height)
        self.boundary_filledge[:,self.width-1] = np.ones(self.height)

    def get_neighbor_coordinate_dis1(self, coordinate, distance=1):
        """Get its eight neighbors' coordinate"""
        # x = coordinate[0]
        # y = coordinate[1]
        neighbors_dict = dict(zip(range(8), [copy.deepcopy(coordinate) for i in range(8)]))
        # position 0
        neighbors_dict[2][0] -= distance  # height
        # position 1 
        neighbors_dict[1][0] -= distance  # height
        neighbors_dict[1][1] += distance  # width
        # position 2
        neighbors_dict[0][1] += distance  # width
        # position 3
        neighbors_dict[7][0] += distance  # height
        neighbors_dict[7][1] += distance  # width
        # position 4
        neighbors_dict[6][0] += distance  # height
        # position 5
        neighbors_dict[5][0] += distance  # height
        neighbors_dict[5][1] -= distance  # width
        # position 6
        neighbors_dict[4][1] -= 1  # width
        # position 7
        neighbors_dict[3][0] -= 1  # height
        neighbors_dict[3][1] -= 1  # width

        return neighbors_dict
    # def get_neighbor_coordinate_dis1(self, coordinate, distance=1):
    #     """Get its eight neighbors' coordinate"""
    #     # x = coordinate[0]
    #     # y = coordinate[1]
    #     neighbors_dict = dict(zip(range(8), [copy.deepcopy(coordinate) for i in range(8)]))
    #     # position 0
    #     neighbors_dict[0][0] -= distance  # height
    #     # position 1 
    #     neighbors_dict[1][0] -= distance  # height
    #     neighbors_dict[1][1] += distance  # width
    #     # position 2
    #     neighbors_dict[2][1] += distance  # width
    #     # position 3
    #     neighbors_dict[3][0] += distance  # height
    #     neighbors_dict[3][1] += distance  # width
    #     # position 4
    #     neighbors_dict[4][0] += distance  # height
    #     # position 5
    #     neighbors_dict[5][0] += distance  # height
    #     neighbors_dict[5][1] -= distance  # width
    #     # position 6
    #     neighbors_dict[6][1] -= 1  # width
    #     # position 7
    #     neighbors_dict[7][0] -= 1  # height
    #     neighbors_dict[7][1] -= 1  # width

    #     return neighbors_dict
    def get_neighbor_coordinate_dis2(self, coordinate, distance=2):
        """Get its eight neighbors' coordinate"""
        # x = coordinate[0]
        # y = coordinate[1]
        neighbors_dict = dict(zip(range(16), [copy.deepcopy(coordinate) for i in range(16)]))
        # position 0
        neighbors_dict[0][0] -= distance  # height
        # position 1 
        neighbors_dict[1][0] -= distance  # height
        neighbors_dict[1][1] += distance  # width
        # position 2
        neighbors_dict[2][1] += distance  # width
        # position 3
        neighbors_dict[3][0] += distance  # height
        neighbors_dict[3][1] += distance  # width
        # position 4
        neighbors_dict[4][0] += distance  # height
        # position 5
        neighbors_dict[5][0] += distance  # height
        neighbors_dict[5][1] -= distance  # width
        # position 6
        neighbors_dict[6][1] -= 1  # width
        # position 7
        neighbors_dict[7][0] -= 1  # height
        neighbors_dict[7][1] -= 1  # width

        # position 8
        neighbors_dict[8][0] -= distance  # height
        neighbors_dict[8][1] -= 1  # width

        # position 9 
        neighbors_dict[9][0] -= distance  # height
        neighbors_dict[9][1] += 1  # width
        # position 10
        neighbors_dict[10][0] -= 1  # height
        neighbors_dict[10][1] += distance  # width
        # position 11
        neighbors_dict[11][0] += 1  # height
        neighbors_dict[11][1] += distance  # width
        # position 12
        neighbors_dict[12][0] += distance  # height
        neighbors_dict[12][1] += 1  # width

        # position 13
        neighbors_dict[13][0] += distance  # height
        neighbors_dict[13][1] -= 1 # width
        # position 14
        neighbors_dict[14][0] += 1  # height
        neighbors_dict[14][1] -= distance  # width
        # position 15
        neighbors_dict[15][0] -= 1  # height
        neighbors_dict[15][1] -= distance  # width

        return neighbors_dict