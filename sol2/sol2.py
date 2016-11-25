import numpy as np

def DFT(signal):
    # TODO - make sure not using van der monde matrix
    if signal.ndim == 1:
        cols = signal.size
        sig = signal[:, np.newaxis]
    else:
        cols = signal.shape[1]
        sig = signal
    logex = (-2 * np.pi * 1j * np.arange(cols)) / cols
    exmat = np.exp(np.matmul(np.arange(cols).reshape(cols, 1), logex[np.newaxis, :]))
    fourier_signal = np.matmul(exmat, sig).transpose()
    return fourier_signal.astype(np.complex128)

    
def IDFT(fourier_signal):
    pass
    # return signal
    
def DFT2(signal):
    pass
    # return fourier_signal
    
def IDFT2(fourier_signa):
    pass
    # return signal

# computes the magnitude of image derivatives
def conv_der(im):
    pass
    # return magnitude

def fourier_der(im):
    pass
    # return magnitude
# TODO - Q1: Why did you get two different magnitude images?

def blur_spatial (im, kernel_size):
    pass
    # return blur_im
    
def blur_fourier (im, kernel_size):
    pass
    # return blur_im
    
'''
Q2: What happens if the center of the gaussian (in the space domain) will not be at the
(0,0) of the image? Why does it happen?
Q3: What is the difference between the two results (Blurring in image space and blurring
in Fourier space)?
'''