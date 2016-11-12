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
def read_image(filename, representation):
    im = imread(filename) # TODO - can i use flatten or mode?   
    if (representation == REP_GREY) & (im.ndim == 2):
        return im.astype(np.float32) / 255
    elif (representation == REP_GREY) & (im.ndim == 3):
        return rgb2gray(im).astype(np.float32)
    elif representation == REP_RGB: # assuming we are not asked to convert grey to rgb
        return im.astype(np.float32) / 255
    else:
        raise Exception('Unsupported representation: {0}'.format(representation))

def imdisplay(filename, representation):
    im = read_image(filename, representation)
    plt.figure()
    if representation == REP_GREY:
        print ('here')
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
    transmat = np.array([0.299, 0.587, 0.114, 0.596, -0.275, -0.321, 0.212, -0.523, 0.311]).reshape(3,3)
    return convert_rep(imRGB, transmat)


def yiq2rgb(imYIQ):
    transmat = np.array([1, 0.956, 0.621, 1, -0.272, -0.647, 1, -1.106, 1.703]).reshape(3,3)
    return convert_rep(imYIQ, transmat)

# return [im_eq, hist_orig, hist_eq]
def histogram_equalize(im_orig):
    if im_orig.ndim == 3:
        # this is a color image- work on the Y axis
        imYIQ = rgb2yiq(im_orig)
        imYIQ[:, :, 0], hist_orig, hist_eq = histogram_equalize(imYIQ[:, :, 0])
        return yiq2rgb(imYIQ), hist_orig, hist_eq
    else:
        # this is an intensity image
        im = (im_orig*255).round().astype(np.uint8)
        hist_orig, bins = np.histogram(im, bins=256)
        hist_cumsum = np.cumsum(hist_orig)
        hist_cumsum_norm = np.round(hist_cumsum * (255 / im.size))
        im_eq = np.interp(im.reshape(1, -1), bins[:-1], hist_cumsum_norm).reshape(im.shape).astype(np.uint8)
        hist_eq = np.histogram(im_eq, bins=256)[0]

        plt.figure()
        plt.subplot(2,2,1)
        plt.imshow(im_orig, cmap=plt.cm.gray)
        plt.subplot(2,2,2)
        plt.imshow(im_eq, cmap=plt.cm.gray)
        plt.subplot(2, 2, 3)
        plt.plot(hist_cumsum, np.arange(256), np.arange(256), np.arange(256))
        plt.subplot(2, 2, 4)
        plt.plot(np.cumsum(hist_eq), np.arange(256), np.arange(256), np.arange(256))
        plt.show()

        return (im_eq.astype(np.float32)/255), hist_orig, hist_eq


# return [im_quant, error]
def quantize (im_orig, n_quant, n_iter):

    # inner function - find the next segment division
    def find_z(hist, hist_cumsum, q=None):
        if q is None:
            # find initial z
            pixs_per_seg = im.size / n_quant
            nz = [np.where(hist_cumsum >= i * pixs_per_seg)[0][0] for i in range(n_quant)]
            nz.append(256)
            return nz
        else:
            nz = np.zeros(n_quant+1)
            nz[1:-1] = np.round((np.mean(np.row_stack((q[1:], q[:-1])), axis=0)))
            nz[-1] = 256
            return nz

    # inner function - find the next quantize values for each segment
    def find_q(zpz, hist, hist_cumsum, z):
        return [np.round(np.sum(zpz[z[zi]:z[zi+1]]) / (hist_cumsum[z[zi+1]-1]-hist_cumsum[z[zi]]+hist[z[zi]])) for zi in range(len(z[:-1]))]

    # find the current error
    def calc_error(hist, z, q):
        return np.sum(hist * np.power(np.arange(256)-calc_color_map(z, q), 2))

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

        im = (im_orig * 255).round().astype(np.uint8)
        hist_orig = np.histogram(im, bins=256)[0]
        hist_cumsum = np.cumsum(hist_orig)
        zpz = hist_orig * np.arange(256)

        # start iterating
        error = []
        q = None
        z = None
        for it in range(n_iter-1):
            new_z = find_z(hist_orig, hist_cumsum, q)
            q = find_q(zpz, hist_orig, hist_cumsum, new_z)
            if (np.all(z==new_z)):
                # converged
                z = new_z
                break
            z = new_z
            error.append(calc_error(hist_orig, z, q))

        im_quant = np.interp(im.reshape(1, -1), np.arange(256), calc_color_map(z,q)).reshape(im.shape).astype(np.float32) / 255
        return im_quant, error

