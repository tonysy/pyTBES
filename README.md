# TBES-python
Texture and Boundary Encoding-based Segmentation

## Dataset

Berkeley Segmentation Dataset: 
http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz


## Method
### Superpixel
I use SLIC method from skimage lib to get superpixel of image

## Run

### Open visdom server
`python -m visdom.server`

### Run for segmentation
`python main.py`

## Todo
- [ ] Add support for non-overlapping
- [ ] Add metric measurement