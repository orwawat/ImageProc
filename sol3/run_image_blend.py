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

    new_height = int(np.round(im1.shape[0] / 64.0) * 64)
    new_width = int(np.round(im1.shape[1] / 64.0) * 64)
    im1 = imresize(im1, (new_height, new_width, im1.ndim)).astype(np.float32) / 255.0
    im2 = imresize(im2, im1.shape).astype(np.float32) / 255.0
    mask[mask <= 0.5] = 0
    mask[mask > 0.5] = 1
    mask = imresize(mask, im1.shape).astype(np.bool)

    blur_ker = 3
    for im_ker in range(3,16,2):
        for depth in range(4,5):
            print('Im:{0}, Mask:{1}, D:{2}'.format(im_ker, blur_ker, depth))
            plt.figure()
            plt.title('Im:{0}, Mask:{1}, D:{2}'.format(im_ker, blur_ker, depth))
            im_blend = sol3.blend_rgb_image(im1, im2, mask, depth, im_ker, blur_ker)
            plt.imshow(im_blend)
    pylab.show(block=True)

if __name__ == '__main__':
    main()
