from scipy.ndimage.filters import convolve
import numpy as np
from scipy.signal import convolve2d

# --------------------- From sol1 ------------------------
from skimage.color import rgb2gray
from scipy.misc import imread

# Constants
REP_GREY = 1
REP_RGB = 2
MIN_INTENSITY = 0
MAX_INTENSITY = 255
CONV_MODE = 'mirror'
RGB2YIQ_MAT = np.array([0.299, 0.587, 0.114, 0.596, -0.275, -0.321, 0.212, -0.523, 0.311],
                       dtype=np.float32).reshape(3, 3)
YIQ2RGB_MAT = np.array([1, 0.956, 0.621, 1, -0.272, -0.647, 1, -1.106, 1.703], dtype=np.float32).reshape(3, 3)

def read_image(filename, representation):
    """
    Reads a given image file and converts it into a given representation
        filename - string containing the image filename to read.
        representation - representation code, either 1 or 2 defining if the output should be either a
                        grayscale image (1) or an RGB image (2).
        return im as np.float32 in range [0,1]
    """
    im = imread(filename)
    if (representation == REP_GREY) & (im.ndim == 2):
        return im.astype(np.float32) / MAX_INTENSITY
    elif (representation == REP_GREY) & (im.ndim == 3):
        return rgb2gray(im).astype(np.float32)
    elif representation == REP_RGB:  # assuming we are not asked to convert grey to rgb
        return im.astype(np.float32) / MAX_INTENSITY
    else:
        raise Exception('Unsupported representation: {0}'.format(representation))

def convert_rep(im, transmat):
    """
    Helper function that takes in a 3d image, and returns a new 3d image of the same shape exactly,
    where for every pixel NewIm[i,j,:]= transmat * Im[i,j,:]

    transmat is a 3*3 tmatrix
    """
    impermuted = np.transpose(im, (2, 0, 1))  # change order of axes so that the main axis is the channels,
    # than rows than columns
    imreshaped = impermuted.reshape(3, -1)
    imconverted = np.matmul(transmat, imreshaped)
    return imconverted.reshape(impermuted.shape).transpose(1, 2, 0).astype(np.float32)



def rgb2yiq(imRGB):
    """
    Transform an RGB image into the YIQ color space
    input and output are float32 in [0,1]
    return image in YIQ
    """
    return convert_rep(imRGB, RGB2YIQ_MAT)



def yiq2rgb(imYIQ):
    """
    Transform an RGB image into the YIQ color space
    input is float32 in [0,1], output is float32 but may not be clipped to range [0,1]
    return image in RGB
    """
    return convert_rep(imYIQ, YIQ2RGB_MAT)  # may not be in [0,1] range!



def clipped_yiq2rgb(imYIQ):
    """
    Helper function to transform yiq to rgb, but clips the result to range [0,1]
    """
    return np.clip(yiq2rgb(imYIQ), a_min=0, a_max=1.0)  # clipping to [0,1]


# ------------------------------------------------------------------------------


# --------------------- From sol2 ------------------------

def blur_spatial(im, kernel_size):
    """
        Performs image blurring using 2D convolution between the image f and a gaussian kernel g

        Inputs:
            im - is the input image to be blurred (grayscale float32 image).
            kernel_Size - is the size of the gaussian kernel in each dimension (an odd integer).

        Output:
            blur_im - is the output blurry image (grayscale float32 image).

    """
    if kernel_size == 1:
        return im
    ker = get_filter_kernel(kernel_size)
    return blur_im(im, ker)

# ------------------------------------------------------------------------------

# --------------------- From sol3 ------------------------

def get_filter_kernel(filter_size):
    """
        Returns a 1d approximation of a gaussian kernel using binomial coefficients.

        Input:
            kernel_size is an ODD number which is the size of the kernel.

        Output:
            ker - a 2d matrix array of type np.float32 (1Xfilter_size matrix). Its sum is 1 and it contains an
                    approximation of a gaussian kernel (with the 0,0 located in the center) in 1d
    """
    if filter_size % 2 == 0:
        raise Exception("Only odd numbers are allowed as kernel sizes")
    ker = np.array([[1]], dtype=np.float32)
    for i in range(filter_size - 1):
        ker = convolve2d(ker, np.array([[1, 1]]))
    return (ker / np.sum(ker)).astype(np.float32)

def blur_im(im, filter):
    """
    Helper function which blurs the given image in both directions with the given filter
    :param im: Image to be blurred
    :param filter:  filter to blurwith
    :return: A blur copy of the image
    """
    blurred_im = convolve(im, filter, mode=CONV_MODE)
    return convolve(blurred_im, filter.transpose(), mode=CONV_MODE).astype(np.float32)

def reduce(im, filter):
    """
    Reduce the size of the image by 2 in each axis by blurring it with the given filter and
    sub-sampling it in even indices.
    :param im: Image to be reduced
    :param filter: The filter to use in the blurring phase
    :return: The reduced image
    """
    return blur_im(im, filter)[::2, ::2]


def expand(im, filter):
    """
    Expands the size of the image by 2 in each axis by padding with 0's in odd indices and blurring it with the given
    filter (times 2 to compensate the loss of illumination)
    :param im: Image to be expanded
    :param filter: The filter to use in the blurring phase
    :return: The expanded image
    """
    expanded_im = np.zeros((im.shape[0] * 2, im.shape[1] * 2), dtype=np.float32)
    expanded_im[::2, ::2] = im.copy()
    return blur_im(expanded_im, filter*2)


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    Construct a Gaussian pyramid of a given image
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter (an odd scalar that represents a squared filter) to be used
                        in constructing the pyramid filter)
    :return: pyr, filter_vec
        pyr: A python array with the pyramid images in it (smaller index-> higher resolution)
        filter_vec: The filter used when blurring the image
    """
    # When input is legal, levels should be equal to max_levels
    levels = min(max_levels, int(np.log2(im.shape[0] // 8)), int(np.log2(im.shape[1] // 8)))

    pyr = [0] * levels
    filter_vec = get_filter_kernel(filter_size)
    pyr[0] = im.copy()
    for lv in range(1, levels):
        pyr[lv] = reduce(pyr[lv - 1], filter_vec)
    return pyr, filter_vec


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    Construct a Laplacian pyramid of a given image
    :param im: a grayscale image with double values in[0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter (an odd scalar that represents a squared filter) to be used
                        in constructing the pyramid filter)
    :return: pyr, filter_vec
        pyr: A python array with the pyramid images in it (smaller index-> higher resolution)
        filter_vec: The filter used when blurring the image
    """
    pyr, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    for lv in range(len(pyr) - 1):
        pyr[lv] -= expand(pyr[lv + 1], filter_vec)
    return pyr, filter_vec


def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    Rebuilds a laplacian pyramid back to a full resolution image. When coeff is all 1's, and filter_vec is the same
            like te one used to build the pyramid, the result will be exactly the original imaage.
    :param lpyr: The Laplacian pyramid generated by build_laplacian_pyramid.
    :param filter_vec: The filter that are generated by build_laplacian_pyramid.
    :param coeff: coeff is a vector. The vector size is the same as the number of levels in the pyramid lpyr.
                Before reconstruction, each level of the pyramid is multiplied in the corresponding coefficient.
    :return: img - The reconstructed image
    """
    weighted_pyr = [lpyr[i] * coeff[i] for i in range(len(lpyr))]
    img = weighted_pyr[-1]
    for i in range(2, len(lpyr) + 1):
        img = expand(img, filter_vec) + weighted_pyr[-i]
    return img


def linear_stretch(im, new_min, new_max):
    """
    Linearly stretch an image to a new range [new_min, new_max]
    :param im: Image to stretch
    :param new_min: minimal value in the stretched image
    :param new_max: maximal value in the stretched image
    :return: The stretched imaage
    """
    old_min, old_max = im.min(), im.max()
    return (((im - old_min) * (new_max - new_min) / (old_max - old_min)) + new_min).astype(np.float32)


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    Blend to grayscale images together.
    :param im1: input grayscale image to be blended.
    :param im2: input grayscale image to be blended. same dimensions as im1
    :param mask: a boolean (i.e. dtype == np.bool) mask containing True and False representing which parts
                    of im1 and im2 should appear in the resulting im_blend. Note that a value of True corresponds to 1,
                    and False corresponds to 0. same dimensions as im1.
    :param max_levels: the max_levels parameter you should use when generating the Gaussian and Laplacian
                        pyramids.
    :param filter_size_im: the size of the Gaussian filter (an odd scalar that represents a squared filter) which
                            defining the filter used in the construction of the Laplacian pyramids of im1 and im2.
    :param filter_size_mask: the size of the Gaussian filter(an odd scalar that represents a squared filter) which
                            defining the filter used in the construction of the Gaussian pyramid of mask.
    :return: im_blend
    """
    if im1.shape != im2.shape or im1.shape != mask.shape or im1.ndim != 2:
        raise Exception('Images and mask size do not match')

    pyr1, filter1 = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    pyr2, filter2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)
    pyr_mask, filter_mask = build_gaussian_pyramid(mask.astype(np.float32), max_levels, filter_size_mask)

    new_pyr = []
    for i in range(min(len(pyr1), len(pyr2), len(pyr_mask))):
        new_pyr.append(np.multiply(pyr1[i], pyr_mask[i]) + np.multiply(pyr2[i], (1 - pyr_mask[i])))
    im_blend = laplacian_to_image(new_pyr, filter1, [1] * (i + 1))
    return im_blend.clip(0, 1).astype(np.float32)


def blend_rgb_image(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    Util function to blend to color images, channel by channel using the blend_image function.
    :param im1: see blend_image
    :param im2: see blend_image
    :param mask:  see blend_image
    :param max_levels:  see blend_image
    :param filter_size_im:  see blend_image
    :param filter_size_mask:  see blend_image
    :return:  see blend_image
    """
    if im1.ndim != 3 or mask.dtype != np.bool or im1.shape[2] != 3:
        raise Exception("Bad usage - image is not rgb or mask is not boolean")

    im_blend = np.zeros(im1.shape)
    for dim in range(im1.shape[2]):
        im_blend[:, :, dim] = pyramid_blending(im1[:, :, dim], im2[:, :, dim], mask, max_levels,
                                               filter_size_im, filter_size_mask)
    return im_blend.astype(np.float32)

# ------------------------------------------------------------------------------
