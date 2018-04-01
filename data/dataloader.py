import os 
import sys 
from PIL import Image 
import numpy as np 

class Dataloader(object):
    """BCD 500 Dataloader"""

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def get_image(self, mode='train'):
        data_path = os.path.join(self.data_dir,'BSDS500/data/images', mode)
        image_data_list = []
        for roots, dirs, files in os.walk(data_path):
            for item in files:
                image_filepath = os.path.join(roots, item)
                assert os.path.exists(image_filepath)

                if os.path.splitext(image_filepath)[1] in ['.png','.jpeg','.jpg']:
                    image = Image.open(image_filepath)
                    # todo: transform
                    image_data_list.append(np.asarray(image))
        # return np.stack(image_data_list)
        
        return image_data_list

    def transform(self):            
        pass 

if __name__ == '__main__':
    bcd500_loader = Dataloader(data_dir='./dataset/BSR')
    train_image = bcd500_loader.get_image(mode='train')
    # import pdb ; pdb.set_trace()
    