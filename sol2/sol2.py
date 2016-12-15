import numpy as np
from numpy.matlib import repmat
from scipy.signal import convolve2d

# --------------------- From sol1 ------------------------
from skimage.color import rgb2gray
from scipy.misc import imread

# Constants
REP_GREY = 1
REP_RGB = 2
MIN_INTENSITY = 0
MAX_INTENSITY = 255


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



def DFT(signal):
    """
        Implemented without loops and compatible with matrices (Extra credit)

        Transform a 1D (or 2d) discrete signal to its Fourier representation
        Input:
            signal is an array of dtype float32 with shape (N,1) (technically it’s 2D)
        Output:
            fourier_signal is an array of dtype complex128 with the same shape.
    """
    if signal.ndim == 1:  # to be able to handle a 1d vector as well
        cols = signal.size
        sig = signal[:, np.newaxis]
    else:
        cols = signal.shape[1]
        sig = signal.transpose()
    logex = (-2 * np.pi * 1j * np.arange(cols)) / cols
    exmat = np.exp(np.matmul(np.arange(cols).reshape(cols, 1), logex[np.newaxis, :]))
    fourier_signal = np.matmul(exmat, sig).transpose()
    if signal.ndim == 1:
        fourier_signal = fourier_signal[0]
    return fourier_signal.astype(np.complex128)


def IDFT(fourier_signal):
    """
        Implemented without loops and compatible with matrices (Extra credit)

        Transform a 1D (or 2d) to its discrete signal from its Fourier representation
        Input:
            fourier_signal is an array of dtype complex128 with shape (N,1)
        Output:
            signal has the same shape and dtype.  Note that when the origin of fourier_signal
                is a transformed real signal you can expect signal to be real valued as well,
                although it may return with a tiny imaginary par
    """
    if fourier_signal.ndim == 1:  # to be able to handle a 1d vector as well
        cols = fourier_signal.size
        sig = fourier_signal[:, np.newaxis]
    else:
        cols = fourier_signal.shape[1]
        sig = fourier_signal.transpose()
    sig = sig / cols
    logex = (2 * np.pi * 1j * np.arange(cols)) / cols
    exmat = np.exp(np.matmul(np.arange(cols).reshape(cols, 1), logex[np.newaxis, :]))
    signal = np.matmul(exmat, sig).transpose()
    if fourier_signal.ndim == 1:
        signal = signal[0]
    return signal


def DFT2(signal):
    """
        Implemented without loops and compatible with matrices

        Converts a 2D discrete signal to its Fourier representation.

        Input:
            image is a grayscale image of dtype float32
        Output:
            and fourier_image is a 2D array of dtype complex128
    """
    if signal.ndim != 2 or (signal.dtype != np.float32 and signal.dtype != np.float64):
        raise Exception("Signal has to be 2d and float32 or float64 to be converted using DFT2")
    xaxisDFT = DFT(signal)
    return DFT(xaxisDFT.transpose()).transpose()


def IDFT2(fourier_signal):
    """
        Implemented without loops and compatible with matrices

        Converts a 2D to a discrete signal from its Fourier representation.

        Input:
            fourier_image is a 2D array of dtype complex128.
        Output:
            image has the same shape and dtype. Note that when the origin of fourier_image is a
                real image transformed with DFT2 you can expect the returned image to be real valued as well,
                although it may return with a tiny imaginary part.
    """
    if fourier_signal.ndim != 2 or fourier_signal.dtype != np.complex128:
        raise Exception("Fourier signal has to be 2d and complex128 to be converted using IDFT2")
    xaxisIDFT = IDFT(fourier_signal)
    return IDFT(xaxisIDFT.transpose()).transpose()


def derive_img(im, axis=0):
    """
        Derives an image in the given axis using simple convolution with [-1, 0 ,1]

        Input:
            a grayscale images of type float32
    """
    if axis != 0:
        return derive_img(im.transpose()).transpose()
    kernel = np.array([[1, 0, -1]], dtype=np.float32)
    return convolve2d(im, kernel, mode='same')


def conv_der(im):
    """
        Computes the magnitude of image derivatives, the image in each using simple convolution with [1, 0, −1]
            as a row and column vectors, to get the two image derivatives.

        Input and Output:
            a grayscale images of type float32
    """
    return np.sqrt(np.power(derive_img(im), 2) + np.power(derive_img(im, 1), 2)).astype(np.float32)


def fourier_der(im):
    """
        Computes the magnitude of image derivatives using Fourier transform, using the
            formula from class to derive in the x and y directions.

        Input and Output:
            a grayscale images of type float32
    """
    im_four_sig = np.fft.fftshift(DFT2(im))
    rows, cols = im.shape

    umat = repmat(np.arange(-(cols // 2), np.ceil(cols / 2.0))[np.newaxis, :], rows, 1)
    vmat = repmat(np.arange(-(rows // 2), np.ceil(rows / 2.0))[:, np.newaxis], 1, cols)

    xder = IDFT2(np.fft.ifftshift(np.multiply(im_four_sig, umat))) * (np.pi * 2j / cols)
    yder = IDFT2(np.fft.ifftshift(np.multiply(im_four_sig, vmat))) * (np.pi * 2j / rows)

    return np.sqrt(np.power(np.abs(xder), 2) + np.power(np.abs(yder), 2)).astype(np.float32)


def get_gaus_ker(kernel_size):
    """
        Returns a 2d approximation of a gaussian kernel using binomial coefficients.

        Input:
            kernel_size is an ODD number which is the size of the kernel in each  direction

        Output:
            ker - a 2d matrix array of type np.float32. Its sum is 1 and it contains an approximation of a gaussian kernel
                (with the 0,0 located in the center)
    """
    if kernel_size % 2 == 0:
        raise Exception("Only odd numbers are allowed as kernel sizes")
    ker = np.array([[1]], dtype=np.float32)
    for i in range(kernel_size - 1):
        ker = convolve2d(ker, np.array([[1, 1]]))
    ker = convolve2d(ker, ker.transpose())
    return ker / np.sum(ker)


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
    ker = get_gaus_ker(kernel_size)
    blur_im = convolve2d(im, ker, mode='same', boundary='wrap')  # wrap to match how the fourier blur works
    return blur_im.astype(np.float32)


def blur_fourier(im, kernel_size):
    """
        performs image blurring with gaussian kernel in Fourier space.

        Inputs:
            im - is the input image to be blurred (grayscale float32 image).
            kernel_Size - is the size of the gaussian kernel in each dimension (an odd integer).

        Output:
            blur_im - is the output blurry image (grayscale float32 image).
    """
    if kernel_size == 1:
        return im

    # Generate a padded gaussian kernel
    padded_ker = np.zeros(im.shape)
    centerx = im.shape[0] // 2
    centery = im.shape[1] // 2
    padded_ker[centerx - kernel_size // 2:centerx + kernel_size // 2 + 1,
    centery - kernel_size // 2:centery + kernel_size // 2 + 1] = \
        get_gaus_ker(kernel_size)

    # Blur the image in the frequency domain
    blur_im = np.multiply(DFT2(im), DFT2(np.fft.ifftshift(padded_ker)))
    return np.real(IDFT2(blur_im)).astype(np.float32)
