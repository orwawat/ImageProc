from sol4_utils import *
from sol4_add import non_maximum_suppression as nms
import numpy as np
from scipy.signal import convolve2d

K = 0.04
BLUR_KER_SIZE = 3
DERIVE_KER = np.array([[1, 0, -1]], dtype=np.float32)

def derive_img(im, axis=0):
    """
        Derives an image in the given axis using simple convolution with [-1, 0 ,1]

        Input:
            a grayscale images of type float32
    """
    if axis != 0:
        return derive_img(im.transpose()).transpose()
    return convolve2d(im, DERIVE_KER, mode='same')


def get_blured_mat_mul(im1, im2):
    return blur_spatial(np.multiply(im1, im2), BLUR_KER_SIZE)

# in my code i should use spread_out_corners (play with n,m but start with n=m=7)
def harris_corner_detector(im):
    """
    Basic harris corner detector (not scale invariant)
    :param im: − grayscale image to find key points inside
    :return: pos - An array with shape (N,2) of [x,y] key points locations in im.
    """
    # Get Ix and Iy with [1,0,-1]
    # blur Ix2, Iy2 and IxIy with blur_spatial kernel 3
    # For each pixel we have M: [[Ix2 IxIy],[IyIx, Iy2]]
    # find R = det(M) − k(trace(M))^2 with k=0.04
    # find response image with R for each pixel
    # use non_maximum_supression to get a binary image with the local maximum points
    # Return the xy coordinates of the corners.
    Ix, Iy, = derive_img(im, 0), derive_img(im, 1)
    Ix2, Iy2, IxIy = get_blured_mat_mul(Ix, Ix), get_blured_mat_mul(Iy, Iy), get_blured_mat_mul(Ix, Iy)
    trace_M = Ix2+Iy2
    det_M = np.multiply(Ix2,Iy2) - np.power(IxIy, 2)
    R = det_M - K*np.power(trace_M,2)
    return np.where(nms(R))