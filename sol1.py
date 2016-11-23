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

# TODO - zpz is not a good name
## TODO - use arange - not for loops!!!
# inner function - find the next quantize values for each segment
def find_q(zpz, hist, hist_cumsum, z):
    # TODO - check no division in 0 here because of merged z!
    lenq = len(z) - 1
    q = np.zeros(lenq)
    if lenq < 1 or z[1] == 0:
        raise Exception("Invalid segments")
    for zi in range(lenq):
        denominator = (hist_cumsum[z[zi + 1] - 1] - hist_cumsum[z[zi]] + hist[z[zi]])
        if denominator == 0:
            print ("fuck!")
            q[zi] = z[zi]
            continue
        nominator = np.sum(zpz[z[zi]:z[zi + 1]])
        q[zi] = np.around(nominator / denominator)
    return q

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


    # find the current error
    def calc_error(hist, z, q):
        return np.sum(hist * np.power(np.arange(256) - calc_color_map(z, q), 2)) # TODO - make sure not sqrt of this

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
    def __init__(self, weight, ranger=(0,256), rangeg=(0,256), rangeb=(0,256)):
        self.weight = weight
        self.ranger = ranger
        self.rangeg = rangeg
        self.rangeb = rangeb

    def median_split_by_long_axis(self, im):
        sliced_im = self.get_sliced_img(im)
        longestaxis = np.argmax((self.ranger[1]-self.ranger[0],self.rangeg[1]-self.rangeg[0], self.rangeb[1]-self.rangeb[0]))
        sliced_im_1d = sliced_im[longestaxis,:]

        if len(sliced_im_1d) == 0:
            print("Wow")

        median_in_axis = int(np.median(sliced_im_1d))
        if longestaxis == 0:
            if median_in_axis == self.ranger[0]:
                median_in_axis += 1
            sliced_weight = np.sum(sliced_im_1d < median_in_axis)
            return RGBBox(sliced_weight, (self.ranger[0], median_in_axis), self.rangeg, self.rangeb), \
                    RGBBox(self.weight - sliced_weight, (median_in_axis, self.ranger[1]), self.rangeg, self.rangeb)
        elif longestaxis == 1:
            if median_in_axis == self.rangeg[0]:
                median_in_axis += 1
            sliced_weight = np.sum(sliced_im_1d < median_in_axis)
            return RGBBox(sliced_weight, self.ranger, (self.rangeg[0], median_in_axis), self.rangeb), \
                   RGBBox(self.weight - sliced_weight, self.ranger, (median_in_axis, self.rangeg[1]), self.rangeb)
        else:
            if median_in_axis == self.rangeb[0]:
                median_in_axis += 1
            sliced_weight = np.sum(sliced_im_1d < median_in_axis)
            return RGBBox(sliced_weight, self.ranger, self.rangeg, (self.rangeb[0], median_in_axis)), \
                   RGBBox(self.weight - sliced_weight, self.ranger, self.rangeg, (median_in_axis, self.rangeb[1]))

    def get_sliced_img(self, im):
        in_r = np.logical_and(im[0, :] >= self.ranger[0], im[0, :] < self.ranger[1])
        in_g = np.logical_and(im[1, :] >= self.rangeg[0], im[1, :] < self.rangeg[1])
        in_b = np.logical_and(im[2, :] >= self.rangeb[0], im[2, :] < self.rangeb[1])
        wherevec = np.logical_and(np.logical_and(in_r, in_g), in_b)
        new_im = np.zeros((3, np.sum(wherevec)))
        new_im[0, :] = im[0, wherevec]
        new_im[1, :] = im[1, wherevec]
        new_im[2, :] = im[2, wherevec]

        if new_im == []:
            print("Wow")


        return new_im

    def get_vol(self):
        return (self.ranger[1] - self.ranger[0]) * (self.rangeg[1] - self.rangeg[0]) * (self.rangeg[1] - self.rangeg[0])


# return [im_quant, error]
def quantize_rgb(im_orig, n_quant, n_iter):
    if im_orig.ndim != 3:
        raise Exception("Can only quantize rgb images")

    im_uint = (im_orig * 255).astype(np.uint8)
    im = np.transpose(im_uint, (2, 0, 1))
    im = im.reshape(3, -1)
    range_im = np.arange(256)
    hist_r = np.histogram(im_uint[:, :, 0], 256)[0]
    hist_cumsum_r, zpz_r = np.cumsum(hist_r), hist_r * range_im
    hist_g = np.histogram(im_uint[:, :, 1], 256)[0]
    hist_cumsum_g, zpz_g = np.cumsum(hist_g), hist_g * range_im
    hist_b = np.histogram(im_uint[:, :, 2], 256)[0]
    hist_cumsum_b, zpz_b = np.cumsum(hist_b), hist_b * range_im

    boxes = [RGBBox(im.size / 3)]
    while len(boxes) < n_quant:
        weights = [int(b.weight) if b.get_vol() > 1 else 0 for b in boxes]
        heviestBoxIdx = np.argmax(weights)
        if weights[heviestBoxIdx] == 0:
            print("Can't divide no more! {0} quants found".format(len(boxes)))
        box = boxes[heviestBoxIdx]
        del boxes[heviestBoxIdx]
        box1, box2 = box.median_split_by_long_axis(im)
        boxes = boxes + [box1, box2]


    print ("Total sum is: {0}, expected sum is: {1}".format(np.sum([b.weight for b in boxes]), im_orig.size/3))
    # now, find the centeroid for each box, and prepare the color map
    # q = np.zeros((1, len(boxes), 3))
    colorMap = np.zeros((256, 256, 256, 3), dtype=np.uint8)
    for box in boxes:
        # for now, not average weight - TODO!
        # print ("Range is: {0},{1},{2}".format(box.ranger, box.rangeg, box.rangeb))
        # print ("Weight is: {0}".format(box.weight))
        #r, g, b = np.around((box.ranger[1]-box.ranger[0]) / 2), np.around((box.rangeg[1]-box.rangeg[0]) / 2), np.around((box.rangeb[1]-box.rangeb[0]) / 2)
        r = find_q(zpz_r, hist_r, hist_cumsum_r, box.ranger)[0]
        g = find_q(zpz_g, hist_g, hist_cumsum_g, box.rangeg)[0]
        b = find_q(zpz_b, hist_b, hist_cumsum_b, box.rangeb)[0]

        # print("R,G,B: {0},{1},{2}".format(r,g,b))
        sliced_colorMap = colorMap[box.ranger[0]:box.ranger[1], box.rangeg[0]:box.rangeg[1], box.rangeb[0]:box.rangeb[1]]
        sliced_colorMap[:, :, :, 0] = r
        sliced_colorMap[:, :, :, 1] = g
        sliced_colorMap[:, :, :, 2] = b
        if r < 0 or g < 0 or b <0 or r > 255 or g > 255 or b > 255:
            print (r,' ',g,' ',b)

    im_quant = np.zeros(im_uint.shape, dtype=np.uint8)
    im_quant[:, :, 0] = colorMap[im_uint[:, :, 0], im_uint[:, :, 1], im_uint[:, :, 2], 0]
    im_quant[:, :, 1] = colorMap[im_uint[:, :, 0], im_uint[:, :, 1], im_uint[:, :, 2], 1]
    im_quant[:, :, 2] = colorMap[im_uint[:, :, 0], im_uint[:, :, 1], im_uint[:, :, 2], 2]
    distmat = np.sqrt(np.sum(np.power(im_orig - im_quant, 2), axis=2))
    error = np.sum(distmat)
    return im_quant.astype(np.float32) / 255, error



'''
README

The quantization procedure needs an initial segment division of [0..255] to segments, z. If a division
will have a gray level segment with no pixels, the procedure will crash (Q1: Why?) - when we find q (specifically here -
 when we find the first q), we find the weighted average of each segment. I.e, we divide by the number of pixel in the
 segment to normalize. since one of the segments have no pixel in it, we in fact divide by 0,
 causing our process to crash

'''

# TODO -staying too much in the same box!