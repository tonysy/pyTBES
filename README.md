# TBES-python
Texture and Boundary Encoding-based Segmentation

## Dataset

Berkeley Segmentation Dataset: 
http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz


## Prepare to run
1. Build im2col python for fast generate texture window representation
- `cd ./lib/im2col/ && python setup.py build_ext --inplace && cd ../../`

## Method
### Superpixel
I use SLIC method from skimage lib to get superpixel of image

