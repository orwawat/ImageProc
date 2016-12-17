import sol3
import os
from scipy.misc import imresize
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab

def main():
    if not os.path.exists('./im1.jpg') or not os.path.exists('./im2.jpg') or not os.path.exists('./mask.jpg'):
        print("files are missing")
        exit(-1)

    im1 = sol3.read_image('./im1.jpg', 2)
    im2 = sol3.read_image('./im2.jpg', 2)
    mask = sol3.read_image('./mask.jpg', 1)


    valid_height = 2 ** int(np.log2(im1.shape[0]))
    valid_width = 2 ** int(np.log2(im1.shape[1]))
    im1 = imresize(im1, (valid_height, valid_width, im1.ndim)).astype(np.float32) / 255.0
    im2 = imresize(im2, im1.shape).astype(np.float32) / 255.0
    mask[mask <= 0.5] = 0
    mask[mask > 0.5] = 1
    mask = imresize(mask, im1.shape).astype(np.bool)
    im_blend = sol3.blend_rgb_image(im1, im2, mask, 8, 5, 3)
    plt.imshow(im_blend)
    pylab.show(block=True)

if __name__ == '__main__':
    main()
