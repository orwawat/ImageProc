import numpy as np
from skimage.color import rgb2gray
from scipy.misc import imread
import matplotlib.pyplot as plt

# Constants

REP_GREY = 1
REP_RGB = 2


# Reads a given image file and converts it into a given representation
# filename - string containing the image filename to read.
# representation - representation code, either 1 or 2 defining if the output should be either a grayscale
# image (1) or an RGB image (2).
# return im
# TODO - is it possible we read float32 in [0,1] already???
# TODO - check input data
# TODO - run tests
# TODO - Document, format and create README
def read_image(filename, representation):
    im = imread(filename)  # TODO - can i use flatten or mode?
    if (representation == REP_GREY) & (im.ndim == 2):
        return im.astype(np.float32) / 255
    elif (representation == REP_GREY) & (im.ndim == 3):
        return rgb2gray(im).astype(np.float32)
    elif representation == REP_RGB:  # assuming we are not asked to convert grey to rgb
        return im.astype(np.float32) / 255
    else:
        raise Exception('Unsupported representation: {0}'.format(representation))


def imdisplay(filename, representation):
    im = read_image(filename, representation)
    plt.figure()
    if representation == REP_GREY:
        print('here')
        plt.imshow(im, cmap=plt.cm.gray)
    else:
        plt.imshow(im)
    plt.axis('off')


def convert_rep(im, transmat):
    impermuted = np.transpose(im, (2, 0, 1))
    imreshaped = impermuted.reshape(3, -1)
    imconverted = np.matmul(transmat, imreshaped)
    return imconverted.reshape(impermuted.shape).transpose(1, 2, 0).astype(np.float32)


# input and output are float32 in [0,1]
def rgb2yiq(imRGB):
    # TODO - can make this matrix a constant
    transmat = np.array([0.299, 0.587, 0.114, 0.596, -0.275, -0.321, 0.212, -0.523, 0.311],
                        dtype=np.float32).reshape(3, 3)
    return convert_rep(imRGB, transmat)


def yiq2rgb(imYIQ):
    transmat = np.array([1, 0.956, 0.621, 1, -0.272, -0.647, 1, -1.106, 1.703], dtype=np.float32).reshape(3, 3)
    converted_im = convert_rep(imYIQ, transmat)
    converted_im = np.clip(converted_im, a_min=0, a_max=1.0)  # clipping to [0,1]
    return converted_im


# return [im_eq, hist_orig, hist_eq]
def histogram_equalize(im_orig):
    if im_orig.ndim == 3:
        # this is a color image- work on the Y axis
        imYIQ = rgb2yiq(im_orig)
        imYIQ[:, :, 0], hist_orig, hist_eq = histogram_equalize(imYIQ[:, :, 0])
        return yiq2rgb(imYIQ), hist_orig, hist_eq
    else:
        # this is an intensity image
        im = (im_orig * 255).around().astype(np.uint8)
        hist_orig, bins = np.histogram(im, bins=256)
        hist_cumsum = np.cumsum(hist_orig)
        hist_cumsum_norm = np.around(hist_cumsum * (255.0 / im.size))  # normalizing and stretching linearly
        # reinterpret image
        im_eq = np.interp(im.reshape(1, -1), bins[:-1], hist_cumsum_norm).reshape(im.shape).astype(np.uint8)
        hist_eq = np.histogram(im_eq, bins=256)[0]

        return (im_eq.astype(np.float32) / 255), hist_orig, hist_eq


# return [im_quant, error]
def quantize(im_orig, n_quant, n_iter):
    # inner function - find the next segment division
    def find_z(hist, hist_cumsum, q=None):
        if q is None:
            # find initial z - equal weighted segments
            pixs_per_seg = im.size / n_quant
            nz = [np.where(hist_cumsum >= i * pixs_per_seg)[0][0] for i in range(n_quant)]
            nz.append(256)
            return nz
        else:
            nz = np.zeros(n_quant + 1)
            nz[1:-1] = np.around((np.mean(np.row_stack((q[1:], q[:-1])), axis=0)))
            nz[-1] = 256
            return nz

    # inner function - find the next quantize values for each segment
    def find_q(zpz, hist, hist_cumsum, z):
        return [np.around(np.sum(zpz[z[zi]:z[zi + 1]]) / (hist_cumsum[z[zi + 1] - 1] - hist_cumsum[z[zi]] + hist[z[zi]]))
                for zi in range(len(z[:-1]))]

    # find the current error
    def calc_error(hist, z, q):
        return np.sum(hist * np.power(np.arange(256) - calc_color_map(z, q), 2))

    def calc_color_map(z, q):
        color_map = np.zeros(256)
        for i in range(n_quant):
            color_map[z[i]:z[i + 1]] = q[i]
        return color_map

    if im_orig.ndim == 3:
        # this is a color image- work on the Y axis
        imYIQ = rgb2yiq(im_orig)
        imYIQ[:, :, 0], error = quantize(imYIQ[:, :, 0], n_quant, n_iter)
        return yiq2rgb(imYIQ), error
    else:
        # this is an intensity image

        im = (im_orig * 255).around().astype(np.uint8)
        hist_orig = np.histogram(im, bins=256)[0]
        hist_cumsum = np.cumsum(hist_orig)
        zpz = hist_orig * np.arange(256)

        # start iterating
        error = []
        q = None
        z = None
        for it in range(n_iter - 1):
            new_z = find_z(hist_orig, hist_cumsum, q)
            q = find_q(zpz, hist_orig, hist_cumsum, new_z)
            if np.all(z == new_z):
                # converged
                z = new_z
                break
            z = new_z
            error.append(calc_error(hist_orig, z, q))

        im_quant = np.interp(im.reshape(1, -1), np.arange(256), calc_color_map(z, q)).reshape(im.shape).astype(
            np.float32) / 255
        return im_quant, error


# What we did in the 1d case was essentially Lloyds alg. for k-means clustering.
# To do the same here, we need to initialize the first centroids somehow, and then repeat the process using
# euclidean distance as our metric.
# Although there are more efficient implementations (like scikit.cluster.kmeans) for clustering (which uses
# randomized centroids and compares several different initializations), I will implement it on my own.
# For initialization (in order to be deterministic), I will use something similar to median-cut in the 3d histogram.
# Also, I will use exhaustive search (and not something more efficient but more complex like Voronoi tessellations)
# return [im_quant, error]
def quantize_rgb(im_orig, n_quant, n_iter):
    # each z - is (r,g,b) vals end of segment starting from last segment
    # initial z - uniform like in 1d case - can start with the heaviest
    pass



'''
README

The quantization procedure needs an initial segment division of [0..255] to segments, z. If a division
will have a gray level segment with no pixels, the procedure will crash (Q1: Why?) - when we find q (specifically here -
 when we find the first q), we find the weighted average of each segment. I.e, we divide by the number of pixel in the
 segment to normalize. since one of the segments have no pixel in it, we in fact divide by 0,
 causing our process to crash

'''