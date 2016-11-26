import numpy as np
from scipy.signal import convolve2d

'''
    Implemented without loops and compatible with matrices
'''
def DFT(signal):
    # TODO - make sure not using van der monde matrix
    if signal.ndim == 1:
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

'''
    Implemented without loops and compatible with matrices
'''
def IDFT(fourier_signal):
    if fourier_signal.ndim == 1:
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
    # TODO - is needed?
    if np.abs(np.imag(signal)).max() < 1e-10: # todo todo todo!!!
        return np.real(signal).astype(np.float32)
    return signal
    # TODO - check if need to return to real


'''
    Implemented without loops and compatible with matrices
'''
def DFT2(signal):
    if signal.ndim != 2 or signal.dtype != np.float32:
        raise Exception("Signal has to be 2d and float32 to be converted using IDFT")
    xaxisDFT = DFT(signal)
    return DFT(xaxisDFT.transpose()).transpose()

'''
    Implemented without loops and compatible with matrices
'''
def IDFT2(fourier_signal):
    if fourier_signal.ndim != 2 or fourier_signal.dtype != np.complex128:
        raise Exception("Fourier signal has to be 2d and complex128 to be converted using IDFT2")
    xaxisIDFT = IDFT(fourier_signal)
    return IDFT(xaxisIDFT.transpose()).transpose()


def derive_img(im, axis=0):
    if axis != 0:
        return derive_img(im.transpose()).transpose()
    kernel = np.array([[-1, 0, 1]], dtype=np.float32)
    return convolve2d(im, kernel, mode='same')

# computes the magnitude of image derivatives
def conv_der(im):
    # todo - should also support 1d?
    return np.sqrt(np.power(derive_img(im), 2) + np.power(derive_img(im, 1), 2))


def fourier_der_image(im, axis=0):
    if axis != 0:
        return fourier_der_image(im.transpose()).transpose()
    kernel = np.array([[-1, 0, 1]], dtype=np.float32)
    return convolve2d(im, kernel, mode='same')

def fourier_der(im):
    # todo - should also support 1d?
    im_four_sig = np.fft.fftshift(DFT2(im))
    rows, cols = im.shape

    umat = np.matmul(np.ones((rows,1)), np.arange(-(cols//2), np.ceil(cols/2.0))[np.newaxis, :])
    vmat = np.matmul(np.arange(-(rows//2), np.ceil(rows/2.0))[:, np.newaxis], np.ones((1, cols)))

    xder = IDFT2(np.fft.ifftshift(np.multiply(im_four_sig, umat))) * (np.pi * 2j / (cols ** 2))
    yder = IDFT2(np.fft.ifftshift(np.multiply(im_four_sig, vmat))) * (np.pi * 2j / (rows ** 2))
    # TODO - what to do with the leftover imaginative parts?
    if np.abs(np.imag(xder)).max() > 1e-4 or np.abs(np.imag(yder)).max() > 1e-4:
        raise Exception("Not cool")
    return np.sqrt(np.power(np.real(xder), 2) + np.power(np.real(yder), 2)).astype(np.float32)
    # return magnitude
# TODO - Q1: Why did you get two different magnitude images?

def get_gaus_ker(kernel_size):
    if kernel_size % 2 == 0:
        raise Exception("Only odd numbers are allowed as kernel sizes")
    ker = np.array([[1]], dtype=np.float32)
    for i in range(kernel_size-1):
        ker = convolve2d(ker, np.array([[1,1]]))
    ker = convolve2d(ker, ker.transpose())
    return ker / np.sum(ker)

def blur_spatial (im, kernel_size):
    if kernel_size == 1:
        return im
    ker = get_gaus_ker(kernel_size)
    blur_im = convolve2d(im, ker, mode='same')
    return blur_im.astype(np.float32)
    
def blur_fourier(im, kernel_size):
    if kernel_size == 1:
        return im
    get_gaus_ker(kernel_size)
    padded_ker = np.zeros(im.shape)
    centerx = im.shape[0] // 2 + 1
    centery = im.shape[1] // 2 + 1
    padded_ker[centerx-kernel_size//2:centerx+kernel_size//2+1, centery-kernel_size//2:centery+kernel_size//2+1] = \
        get_gaus_ker(kernel_size)
    blur_im = np.multiply(DFT2(im), np.fft.ifftshift(padded_ker))
    return np.real(IDFT2(blur_im)).astype(np.float32)  # TODO - what to do with this conversion?

    
'''
Q2: What happens if the center of the gaussian (in the space domain) will not be at the
(0,0) of the image? Why does it happen?
Q3: What is the difference between the two results (Blurring in image space and blurring
in Fourier space)?
'''