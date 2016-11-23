# TODO - check input data
# TODO - run tests
# TODO - check in quantization division in 0
# TODO - remove prints
# TODO - check against avichai
# TODO - make sure error is squared
# TODO  - look at submission guidlines - make sure no limit on chars
# TODO - renames!!
# TODO - constants
## TODO - use arange - not for loops!!!
# TODO - make the bonus question more efficient
# TODO - go over the questions to make sure I addressed everything
import numpy as np
from skimage.color import rgb2gray
from scipy.misc import imread
import matplotlib.pyplot as plt

# Constants
REP_GREY = 1
REP_RGB = 2
RGB2YIQ_MAT = np.array([0.299, 0.587, 0.114, 0.596, -0.275, -0.321, 0.212, -0.523, 0.311],
                       dtype=np.float32).reshape(3, 3)
YIQ2RGB_MAT = np.array([1, 0.956, 0.621, 1, -0.272, -0.647, 1, -1.106, 1.703], dtype=np.float32).reshape(3, 3)

'''
    Reads a given image file and converts it into a given representation
        filename - string containing the image filename to read.
        representation - representation code, either 1 or 2 defining if the output should be either a grayscale
                        image (1) or an RGB image (2).
        return im as np.float32 in range [0,1]
'''


def read_image(filename, representation):
    im = imread(filename)
    if (representation == REP_GREY) & (im.ndim == 2):
        return im.astype(np.float32) / 255
    elif (representation == REP_GREY) & (im.ndim == 3):
        return rgb2gray(im).astype(np.float32)
    elif representation == REP_RGB:  # assuming we are not asked to convert grey to rgb
        return im.astype(np.float32) / 255
    else:
        raise Exception('Unsupported representation: {0}'.format(representation))


'''
    utilizes read_image to display a given image file in a given representation.
    opens a new figure for that.
    filename - string containing the image filename to read.
    representation - representation code, either 1 or 2 defining if the output should be either a grayscale
                        image (1) or an RGB image (2).
'''


def imdisplay(filename, representation):
    im = read_image(filename, representation)
    plt.figure()
    if representation == REP_GREY:
        plt.imshow(im, cmap=plt.cm.gray)
    else:
        plt.imshow(im)
    plt.axis('off')
    plt.show()


'''
    Helper function that takes in a 3d image, and returns a new 3d image of the same shape exactly,
    where for every pixel NewIm[i,j,:]= transmat * Im[i,j,:]

    transmat is a 3*3 tmatrix
'''


def convert_rep(im, transmat):
    impermuted = np.transpose(im, (2, 0, 1))  # change order of axes so that the main axis is the channels,
    # than rows than columns
    imreshaped = impermuted.reshape(3, -1)
    imconverted = np.matmul(transmat, imreshaped)
    return imconverted.reshape(impermuted.shape).transpose(1, 2, 0).astype(np.float32)


'''
    Transform an RGB image into the YIQ color space
    input and output are float32 in [0,1]
    return image in YIQ
'''


def rgb2yiq(imRGB):
    return convert_rep(imRGB, RGB2YIQ_MAT)


'''
    Transform an RGB image into the YIQ color space
    input is float32 in [0,1], output is float32 but may not be clipped to range [0,1]
    return image in RGB
'''


def yiq2rgb(imYIQ):
    return convert_rep(imYIQ, YIQ2RGB_MAT)  # may not be in [0,1] range!


'''
    Helper function to transform yiq to rgb, but clips the result to range [0,1]
'''


def clipped_yiq2rgb(imYIQ):
    return np.clip(yiq2rgb(imYIQ), a_min=0, a_max=1.0)  # clipping to [0,1]

'''
    Performs histogram equalization of a given grayscale or RGB image
    If an RGB image is given, the following equalization procedure only operate on the Y channel of
    the corresponding YIQ image and then convert back from YIQ to RGB. Moreover, the outputs hist_orig
    and hist_eq are the histogram of the Y channel only.
    im_orig - is the input grayscale or RGB float32 image with values in [0, 1].

    Returns:
        im_eq - is the equalized image. grayscale or RGB float32 image with values in [0, 1].
        hist_orig - is a 256 bin histogram of the original image (array with shape (256,) ).
        hist_eq - is a 256 bin histogram of the equalized image (array with shape (256,) ).
'''


def histogram_equalize(im_orig):
    if im_orig.ndim == 3:
        # this is a color image - work on the Y axis
        imYIQ = rgb2yiq(im_orig)
        imYIQ[:, :, 0], hist_orig, hist_eq = histogram_equalize(imYIQ[:, :, 0])
        return clipped_yiq2rgb(imYIQ), hist_orig, hist_eq
    else:
        # this is an intensity image
        im = np.around(im_orig * 255).astype(np.uint8)
        hist_orig, bins = np.histogram(im, bins=256, range=[0, 255])  # TODO - make sure doesn't suppose to be 255
        hist_cumsum = np.cumsum(hist_orig)
        hist_cumsum_norm = hist_cumsum * (255.0 / im.size)  # normalizing the cumsum

        # stretch if needed: (to have color map stretch between 0 to 255)
        # this is a monotonically increasing function, which is cumulative distribution so
        # hist_cumsum_norm[-1]=255 and min(hist_cumsum_norm)=hist_cumsum_norm[0
        if hist_cumsum_norm[0] != 0 or hist_cumsum_norm[-1] != 255:
            cm = hist_cumsum_norm[0]  # minimal value in normed_cumsum which should be stretched to 0
            hist_cumsum_norm = ((hist_cumsum_norm - cm) * 255) / (hist_cumsum_norm[-1] - cm)
            if (hist_cumsum_norm.min() < 0):
                raise Exception("Error in stretching the normalized histogram")
        hist_cumsum_norm = np.around(hist_cumsum_norm)

        # reinterpret image
        im_eq = (hist_cumsum_norm[im]).astype(np.uint8)
        hist_eq = np.histogram(im_eq, bins=256, range=[0, 255])[0]

        return (im_eq.astype(np.float32) / 255), hist_orig, hist_eq


'''
    Inner function - find the next quantize values for each segment

    Inputs:
        zpz is the weighted histogram - the histogram of the image multiplied (per element) in the value of the bin
        hist is the original histogram
        his_cumsum - the cumsum of the hist
        z - the segments in which to find q's. array like

    Outputs:
        q - array like of length len(z)-1, q[0] is the q in segment (z[0]:z[1]) etc.
'''
def find_q(zpz, hist, hist_cumsum, z):
    lenq = len(z) - 1
    q = np.zeros(lenq)
    if lenq < 1 or z[1] == 0:
        raise Exception("Invalid segments")
    for zi in range(lenq):
        denominator = (hist_cumsum[z[zi + 1] - 1] - hist_cumsum[z[zi]] + hist[z[zi]])
        if denominator == 0:  # To avoid division in zero when segments converge
            q[zi] = z[zi]
            continue
        nominator = np.sum(zpz[z[zi]:z[zi + 1]])
        q[zi] = np.around(nominator / denominator)
    return q

'''
    Helper function to find segments from given centroids (1d).
    If centroids are None, return segments s.t for each segment, the number of pixels is (approx.)
    total_num_pixels/n_quants

    Inputs:
        n_quant - number of quants
        hist_cumsum - the histogram cumsum
        q - the centroids. array like

    Outputs:
        z - array like of length n_quant + 1. z[0]=0,z[-1]=256 always. z is monotonically increases.
            represents the segments in the histogram
'''
def find_z(n_quant, hist_cumsum, q=None):
    if q is None:
        # find initial z - equal weighted segments
        pixs_per_seg = hist_cumsum[-1] / n_quant
        nz = [0] + [np.where(hist_cumsum <= (i + 1) * pixs_per_seg)[0][-1] for i in range(n_quant)]
        nz[-1] = 256
        return nz
    else:
        nz = np.zeros(n_quant + 1)
        nz[1:-1] = np.around((np.mean(np.row_stack((q[1:], q[:-1])), axis=0)))
        nz[-1] = 256
        return nz.astype(np.uint32)

'''
    Performs optimal quantization of a given grayscale or RGB image
    If an RGB image is given, the following quantization procedure only operate on the Y channel of
    the corresponding YIQ image and then convert back from YIQ to RGB.

    Inputs:
        im_orig - is the input grayscale or RGB image to be quantized (float32 image with values in [0, 1]).
        n_quant - is the number of intensities your output im_quant image should have.
        n_iter - is the maximum number of iterations of the optimization procedure (may converge earlier.)

    Outputs:
        im_quant - is the quantize output image.
        error - is an array with shape (n_iter,) (or less) of the total intensities error for each iteration in the
                   quantization procedure.

'''
def quantize(im_orig, n_quant, n_iter):

    # find the current error (from segment division and centroids)
    def calc_error(hist, z, q):
        return np.sum(hist * np.power(np.arange(256) - calc_color_map(z, q), 2))  # TODO - make sure not sqrt of this

    # inner function - get color map from segments division and centroids
    def calc_color_map(z, q):
        color_map = np.zeros(256)
        for i in range(n_quant):
            color_map[z[i]:z[i + 1]] = q[i]
        return color_map

    if im_orig.ndim == 3:
        # this is a color image- work on the Y axis
        imYIQ = rgb2yiq(im_orig)
        imYIQ[:, :, 0], error = quantize(imYIQ[:, :, 0], n_quant, n_iter)
        return yiq2rgb(imYIQ), error  # Not clipped as instructed!
    else:
        # this is an intensity image
        im = np.around(im_orig * 255).astype(np.uint8)
        hist_orig = np.histogram(im, bins=256, range=[0, 255])[0]
        hist_cumsum = np.cumsum(hist_orig)
        zpz = hist_orig * np.arange(256)

        # start iterating
        error = []
        q = None
        z = None
        for it in range(n_iter):
            new_z = find_z(n_quant, hist_cumsum, q)
            q = find_q(zpz, hist_orig, hist_cumsum, new_z)
            if np.all(z == new_z):
                # converged
                z = new_z
                break
            z = new_z
            error.append(calc_error(hist_orig, z, q))

        cmap = calc_color_map(z, q)
        im_quant = (cmap[im]).astype(np.float32) / 255
        # im_quant = np.interp(im.reshape(1, -1), np.arange(256), calc_color_map(z, q)).reshape(im.shape).astype(
        #     np.float32) / 255
        return im_quant, np.asarray(error).astype(np.float32)



# ---------------------------------------------------------------------------------
# ---------------------------- Extra Credit section -------------------------------
# ---------------------------------------------------------------------------------
'''
Notes:
    What we did in the 1d case was essentially Lloyds alg. for k-means clustering.
    To do the same here, we need to initialize the first centroids somehow, and then repeat the process using
    euclidean distance as our metric.
    Although there are more efficient implementations (like scikit.cluster.kmeans) for clustering (which uses
    randomized centroids and compares several different initializations), I will implement it on my own.
    For initialization (in order to be deterministic), I will use something similar to median-cut in the 3d histogram.
    Also, I will use exhaustive search (and not something more efficient but more complex like Voronoi tessellations)
'''


'''
    TODO
'''
class RGBBox:

    '''
        TODO
    '''
    def __init__(self, weight, ranger=(0, 256), rangeg=(0, 256), rangeb=(0, 256)):
        self.weight = weight
        self.ranger = ranger
        self.rangeg = rangeg
        self.rangeb = rangeb

    '''
    TODO
    '''
    def median_split_by_long_axis(self, im):
        sliced_im = self.get_sliced_img(im)
        longestaxis = np.argmax(
            (self.ranger[1] - self.ranger[0], self.rangeg[1] - self.rangeg[0], self.rangeb[1] - self.rangeb[0]))
        sliced_im_1d = sliced_im[longestaxis, :]

        if len(sliced_im_1d) == 0:
            raise Exception("Error in median split - an empty image channel 1d received")

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

    '''
        TODO
    '''
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
            raise Exception("Failed to get sliced image - an empty image is received")

        return new_im

    '''
        TODO
    '''
    def get_vol(self):
        return (self.ranger[1] - self.ranger[0]) * (self.rangeg[1] - self.rangeg[0]) * (self.rangeg[1] - self.rangeg[0])


# return [im_quant, error]
'''
    TODO
'''
def quantize_rgb(im_orig, n_quant, n_iter):
    if im_orig.ndim != 3:
        raise Exception("Can only quantize rgb images")
    if n_quant > (256 ** 3) / 2:
        raise Exception("Too many quants -can't quantize")

    im_uint = (im_orig * 255).astype(np.uint8)
    im = np.transpose(im_uint, (2, 0, 1))
    im = im.reshape(3, -1)
    range_im = np.arange(256)
    hist_r = np.histogram(im_uint[:, :, 0], 256, range=[0, 255])[0]
    hist_cumsum_r, zpz_r = np.cumsum(hist_r), hist_r * range_im
    hist_g = np.histogram(im_uint[:, :, 1], 256, range=[0, 255])[0]
    hist_cumsum_g, zpz_g = np.cumsum(hist_g), hist_g * range_im
    hist_b = np.histogram(im_uint[:, :, 2], 256, range=[0, 255])[0]
    hist_cumsum_b, zpz_b = np.cumsum(hist_b), hist_b * range_im

    boxes = [RGBBox(im.size / 3)]
    min_box_vol2split = (256 ** 3) / n_quant
    min_weight2split = im.size / (3 * n_quant)
    while len(boxes) < n_quant:
        while True:
            weights = [int(b.weight) if (b.get_vol() > min_box_vol2split and b.weight > min_weight2split) else 0 for b
                       in boxes]
            heviestBoxIdx = np.argmax(weights)
            if weights[heviestBoxIdx] == 0:
                # TODO - comment "dividing min size"
                min_box_vol2split = np.round(min_box_vol2split / 2)
                min_weight2split = np.round(min_weight2split / 2)
                if min_box_vol2split == 1:
                    # TODO - comment - "Can't divide no more! {0} quants found".format(len(boxes)))
                    heviestBoxIdx = -1
                    break
            else:
                break
        if heviestBoxIdx < 0:
            break
        box = boxes[heviestBoxIdx]
        del boxes[heviestBoxIdx]
        box1, box2 = box.median_split_by_long_axis(im)
        boxes = boxes + [box1, box2]

    # now, find the centeroid for each box, and prepare the color map
    colorMap = np.zeros((256, 256, 256, 3), dtype=np.uint8)
    for box in boxes:
        r = find_q(zpz_r, hist_r, hist_cumsum_r, box.ranger)[0]
        g = find_q(zpz_g, hist_g, hist_cumsum_g, box.rangeg)[0]
        b = find_q(zpz_b, hist_b, hist_cumsum_b, box.rangeb)[0]

        sliced_colorMap = colorMap[box.ranger[0]:box.ranger[1], box.rangeg[0]:box.rangeg[1],
                          box.rangeb[0]:box.rangeb[1]]
        sliced_colorMap[:, :, :, 0] = r
        sliced_colorMap[:, :, :, 1] = g
        sliced_colorMap[:, :, :, 2] = b
        if r < 0 or g < 0 or b < 0 or r > 255 or g > 255 or b > 255:
            raise Exception("Failed to quantize - got illegal r,g,b values")

    im_quant = np.zeros(im_uint.shape, dtype=np.uint8)
    im_quant[:, :, 0] = colorMap[im_uint[:, :, 0], im_uint[:, :, 1], im_uint[:, :, 2], 0]
    im_quant[:, :, 1] = colorMap[im_uint[:, :, 0], im_uint[:, :, 1], im_uint[:, :, 2], 1]
    im_quant[:, :, 2] = colorMap[im_uint[:, :, 0], im_uint[:, :, 1], im_uint[:, :, 2], 2]
    distmat = np.sqrt(np.sum(np.power(im_orig - im_quant, 2), axis=2))
    error = np.sum(distmat)
    return im_quant.astype(np.float32) / 255, error
