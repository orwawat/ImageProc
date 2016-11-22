import numpy as np
from skimage.color import rgb2gray
from scipy.misc import imread
import matplotlib.pyplot as plt
from scipy.stats.mstats import mquantiles
from math import log2

# Constants
REP_GREY = 1
REP_RGB = 2
RGB2YIQ_MAT = np.array([0.299, 0.587, 0.114, 0.596, -0.275, -0.321, 0.212, -0.523, 0.311],
                       dtype=np.float32).reshape(3, 3)
YIQ2RGB_MAT = np.array([1, 0.956, 0.621, 1, -0.272, -0.647, 1, -1.106, 1.703], dtype=np.float32).reshape(3, 3)


# Reads a given image file and converts it into a given representation
# filename - string containing the image filename to read.
# representation - representation code, either 1 or 2 defining if the output should be either a grayscale
# image (1) or an RGB image (2).
# return im
# TODO - is it possible we read float32 in [0,1] already???
# TODO - check input data
# TODO - run tests
# TODO - Document, format and create README
# TODO - wrap in try catch?
# TODO - check in quantization division in 0
def read_image(filename, representation):
    im = imread(filename)  # TODO - can i use flatten or mode?
    if (representation == REP_GREY) & (im.ndim == 2):
        return im.astype(np.float32) / 255
    elif (representation == REP_GREY) & (im.ndim == 3):
        return rgb2gray(im).astype(np.float32) # TODO - test if is in [0,1]
    elif representation == REP_RGB:  # assuming we are not asked to convert grey to rgb
        return im.astype(np.float32) / 255
    else:
        raise Exception('Unsupported representation: {0}'.format(representation))


def imdisplay(filename, representation):
    im = read_image(filename, representation)
    plt.figure()
    if representation == REP_GREY:
        plt.imshow(im, cmap=plt.cm.gray)
    else:
        plt.imshow(im)
    plt.axis('off')
    plt.show()


def convert_rep(im, transmat):
    impermuted = np.transpose(im, (2, 0, 1))
    imreshaped = impermuted.reshape(3, -1)
    imconverted = np.matmul(transmat, imreshaped)
    return imconverted.reshape(impermuted.shape).transpose(1, 2, 0).astype(np.float32)


# input and output are float32 in [0,1]
def rgb2yiq(imRGB):
    return convert_rep(imRGB, RGB2YIQ_MAT)


def yiq2rgb(imYIQ):
    return convert_rep(imYIQ, YIQ2RGB_MAT)  # may not be in [0,1] range!


def clipped_yiq2rgb(imYIQ):
    return np.clip(yiq2rgb(imYIQ), a_min=0, a_max=1.0)  # clipping to [0,1]


# return [im_eq, hist_orig, hist_eq]
def histogram_equalize(im_orig):
    if im_orig.ndim == 3:
        # this is a color image- work on the Y axis
        imYIQ = rgb2yiq(im_orig)
        imYIQ[:, :, 0], hist_orig, hist_eq = histogram_equalize(imYIQ[:, :, 0])
        return clipped_yiq2rgb(imYIQ), hist_orig, hist_eq
    else:
        # this is an intensity image
        im = (im_orig * 255).around().astype(np.uint8)
        hist_orig, bins = np.histogram(im, bins=256)
        hist_cumsum = np.cumsum(hist_orig)
        hist_cumsum_norm = np.around(hist_cumsum * (255.0 / im.size))  # normalizing and stretching linearly

        # stretch if needed: TODO - test
        if hist_cumsum_norm[0] != 0 or hist_cumsum_norm[-1] != 255:  # in fact, hist_cumsum_norm[-1] always is 255
            cm = np.where(hist_cumsum_norm > 0)[0]  # first index of hist_cumsum_norm that is not 0
            hist_cumsum_norm = np.around(((hist_cumsum_norm - cm) * 255) / (hist_cumsum_norm[-1] - cm))

        # reinterpret image
        im_eq = np.interp(im.reshape(1, -1), bins[:-1], hist_cumsum_norm).reshape(im.shape).astype(np.uint8)
        hist_eq = np.histogram(im_eq, bins=256)[0]

        return (im_eq.astype(np.float32) / 255), hist_orig, hist_eq  # TODO - make sure still float32!


# return [im_quant, error]
def quantize(im_orig, n_quant, n_iter):
    # inner function - find the next segment division
    def find_z(n_quant, hist_cumsum, q=None):
        if q is None:
            other_z = mquantiles(hist_cumsum, [1.0 / n_quant] * n_quant) # test to see it works the same
            # find initial z - equal weighted segments
            pixs_per_seg = im.size / n_quant # TODO - make sure not int division
            nz = [np.where(hist_cumsum >= i * pixs_per_seg)[0][0] for i in range(n_quant)]
            nz.append(256)
            return nz
        else:
            nz = np.zeros(n_quant + 1)
            nz[1:-1] = np.around((np.mean(np.row_stack((q[1:], q[:-1])), axis=0)))
            nz[-1] = 256
            return nz
# TODO - zpz is not a good name
## TODO - use arange - not for loops!!!
    # inner function - find the next quantize values for each segment
    def find_q(zpz, hist, hist_cumsum, z):
        # TODO - check no division in 0 here because of merged z!
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
        return clipped_yiq2rgb(imYIQ), error
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
            new_z = find_z(n_quant, hist_cumsum, q)
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

# TODO - median cut
'''
What we did in the 1d case was essentially Lloyds alg. for k-means clustering.
To do the same here, we need to initialize the first centroids somehow, and then repeat the process using
euclidean distance as our metric.
Although there are more efficient implementations (like scikit.cluster.kmeans) for clustering (which uses
randomized centroids and compares several different initializations), I will implement it on my own.
For initialization (in order to be deterministic), I will use something similar to median-cut in the 3d histogram.
Also, I will use exhaustive search (and not something more efficient but more complex like Voronoi tessellations)
'''


class RGBBox:
    def __init__(self, ranger=(0,256), rangeg=(0,256), rangeb=(0,256)):
        self.ranger = ranger
        self.rangeg = rangeg
        self.rangeb = rangeb
    # TODO - add total weight and then use the weight to decide who to divide
    def median_split_by_long_axis(self, im):
        sliced_im = self.get_sliced_img(im)
        longestaxis = np.argmax((self.ranger[1]-self.ranger[0],self.rangeg[1]-self.rangeg[0], self.rangeb[1]-self.rangeb[0]))
        median_in_axis = np.median(sliced_im[longestaxis,:])
        if longestaxis == 0:
            return RGBBox((self.ranger[0], median_in_axis), self.rangeb, self.rangeb), \
                    RGBBox((median_in_axis, self.ranger[1]), self.rangeb, self.rangeb)
        elif longestaxis == 1:
            return RGBBox(self.ranger, (self.rangeע[0], median_in_axis), self.rangeb), \
                   RGBBox(self.ranger, (median_in_axis, self.rangeע[1]), self.rangeb)
        else:
            return RGBBox(self.ranger, self.rangeb, (self.rangeb[0], median_in_axis)), \
                   RGBBox(self.ranger, self.rangeb, (median_in_axis, self.rangeb[1]))

    def get_sliced_img(self, im):
        in_r = np.logical_and(im[0, :] >= self.ranger[0], im[0, :] < self.ranger[1])
        in_g = np.logical_and(im[1, :] >= self.rangeg[0], im[1, :] < self.rangeg[1])
        in_b = np.logical_and(im[2, :] >= self.rangeb[0], im[2, :] < self.rangeb[1])
        wherevec = np.logical_and(np.logical_and(in_r, in_g), in_b)
        new_im = np.zeros((3, np.sum(wherevec)))
        new_im[0, :] = im[0, wherevec]
        new_im[1, :] = im[1, wherevec]
        new_im[2, :] = im[2, wherevec]
        return new_im


# return [im_quant, error]
def quantize_rgb(im_orig, n_quant, n_iter):
    im = (im_orig * 255).astype(np.uint8)
    im = np.transpose(im, (2, 0, 1))
    im = im.reshape(3, -1)
    #hist_r, hist_g, hist_b = np.histogram(im[:,:,0], 256)[0], np.histogram(im[:,:,1], 256)[0], np.histogram(im[:,:,2], 256)[0]

    boxes = [RGBBox()]
    while len(boxes) < np.ceil(log2(n_quant)):
        newboxes = []
        for box in boxes:
            box1, box2 = box.median_split_by_long_axis(im)
            newboxes.append([box1, box2])
        boxes = newboxes

    # now merge if there are too many

    # start iterating
    error = []
    centroids = None
    segments = None
    for it in range(n_iter - 1):
        new_segments = find_new_segments(n_quant, hist3d, centroids)
        centroids = find_new_centroids(hist3d, segments)
        if np.all(segments == new_segments):
            # converged
            segments = new_segments
            break
        segments = new_segments
        error.append(calc_error(hist3d, segments, centroids))

    im_quant = find_color_map(segments, centroids)[im].astype(np.float32) / 255
    return im_quant, error



'''
README

The quantization procedure needs an initial segment division of [0..255] to segments, z. If a division
will have a gray level segment with no pixels, the procedure will crash (Q1: Why?) - when we find q (specifically here -
 when we find the first q), we find the weighted average of each segment. I.e, we divide by the number of pixel in the
 segment to normalize. since one of the segments have no pixel in it, we in fact divide by 0,
 causing our process to crash

'''