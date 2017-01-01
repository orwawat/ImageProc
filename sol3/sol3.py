from scipy.ndimage.filters import convolve
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import os

# --------------------- From sol1 ------------------------
from skimage.color import rgb2gray
from scipy.misc import imread

# Constants
REP_GREY = 1
REP_RGB = 2
MIN_INTENSITY = 0
MAX_INTENSITY = 255
CONV_MODE = 'reflect'


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


# ------------------------------------------------------------------------------


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
    if im.shape[0] % 2**(levels-1) != 0 or im.shape[1] % 2**(levels-1):
        raise Exception("image shape doesn't meet prerequisites")

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


def render_pyramid(pyr, levels):
    """
    Renders a new image with the first 'levels' number of images from the pyramid.
    :param pyr: either a Gaussian or Laplacian pyramid as defined above.
    :param levels: the number of levels to present in the result ≤ max_levels.
    :return: res - a single black image in which the pyramid levels of the given pyramid pyr are stacked
                    horizontally (after stretching the values to [0, 1])
    """
    width = np.sum([pyr[i].shape[1] for i in range(levels)])
    height = pyr[0].shape[0]
    res = np.zeros((height, width), dtype=np.float32)
    cur_col = 0
    for i in range(levels):
        end_col = cur_col + pyr[i].shape[1]
        res[:pyr[i].shape[0], cur_col:end_col] = linear_stretch(pyr[i], 0., 1.)
        cur_col = end_col
    return res


def display_pyramid(pyr, levels):
    """
    Renders 'levels' number of the pyramid images and displays them.
    :param pyr: either a Gaussian or Laplacian pyramid as defined above.
    :param levels: the number of levels to present in the result ≤ max_levels.
    :return: None
    """
    res = render_pyramid(pyr, levels)
    plt.figure()
    if pyr[0].ndim == 2:  # Gray image
        plt.imshow(res, cmap=plt.cm.gray)
    else:  # Color image
        plt.imshow(res)
    plt.show()


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
        im_blend[:, :, dim] = pyramid_blending(im1[:, :, dim], im2[:, :, dim], mask, max_levels, filter_size_im,
                                               filter_size_mask)
    return im_blend.astype(np.float32)


def load_mask(mask_pth):
    """
    Reads the given mask as a grey level image, converting it to a boolean array
    :param mask_pth: Path of the mask
    :return: ndarray of type bool, same shape as the image read from the path
    """
    mask = read_image(mask_pth, REP_GREY)
    return mask == 1.


def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)


def show_example(title, im1, im2, mask, im_blend):
    """
    Given example images, shows the result in a new figure
    :param title: The title of the figure
    :param im1: The first image (RGB image)
    :param im2: The second image (RGB image)
    :param mask: The mask used in the blending (type bool)
    :param im_blend: The blended image
    """
    fig = plt.figure()
    fig.canvas.set_window_title(title)
    plt.suptitle(title)
    plt.subplot(2, 2, 1)
    plt.imshow(im1)
    plt.title('Im1')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(im2)
    plt.title('Im2')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(mask, cmap=plt.cm.gray)
    plt.title('mask')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(im_blend)
    plt.title('im_blend')
    plt.axis('off')

    plt.show()


def blending_example1():
    """
    Load two images and a mask (butterfly-elephant), blend them, shows the result and return
    :return: im1, im2, mask, im_blend
    """
    im1 = read_image(relpath('externals/baby_elephant.jpg'), 2)
    im2 = read_image(relpath('externals/butterfly.jpg'), 2)
    mask = load_mask(relpath('externals/elephant_butterfly_mask.png'))
    im_blend = blend_rgb_image(im1, im2, mask, 3, 3, 3)
    show_example('Example 1: Baby-Elephant', im1, im2, mask, im_blend)
    return im1, im2, mask, im_blend


def blending_example2():
    """
    Load two images and a mask (SnowboarDog), blend them, shows the result and return
    :return: im1, im2, mask, im_blend
    """
    im1 = read_image(relpath('externals/snowboarder.jpg'), 2)
    im2 = read_image(relpath('externals/dog.jpg'), 2)
    mask = load_mask(relpath('externals/snowboarder_dog_mask.png'))
    im_blend = blend_rgb_image(im1, im2, mask, 6, 5, 11)
    show_example('Example 2: SnowboarDog', im1, im2, mask, im_blend)
    return im1, im2, mask, im_blend
