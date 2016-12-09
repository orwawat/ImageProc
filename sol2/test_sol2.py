import numpy as np
import sol1
import sol2
import matplotlib.pyplot as plt
from matplotlib import pylab
from scipy.ndimage.filters import gaussian_filter
import os

EPSILON = 1e-7

image_pth = r'C:\Users\Maor\Documents\ImageProc\current\external'
#image_pth = r'/cs/usr/maor_i/safe/PycharmProjects/ImageProc/sol2/external'
images = [os.path.join(image_pth, 'jerusalem.jpg'),
            os.path.join(image_pth, 'Low_Contrast.jpg'),
            os.path.join(image_pth, 'monkey.jpg'),
            os.path.join(image_pth, 'head-shot.jpg'),
            os.path.join(image_pth, 'jerusalem.jpg')]
images_grey = images + [os.path.join(image_pth, 'head-shot_grey.jpg')]

def stretch(im, mn=0, mx=1):
    return ((mx-mn)*(im-im.min()))/(im.max()- im.min())

# test dft
# grey read
ffts = []
def test_dft():
    print("Testing 1d DFT...")
    for i, r in enumerate([np.cos(np.arange(100)), np.sin(np.arange(100)), np.power(np.arange(100), 2), np.arange(100)]):
        im = stretch(r)
        npfft = np.fft.fft(im)
        myfft = sol2.DFT(im)
        if myfft.dtype != np.complex128:
            raise Exception("Failed in DFT - returned wrong type")
        if not np.all(npfft == myfft):
            maxdiff = np.absolute(npfft - myfft).max()
            if maxdiff > EPSILON:
                raise Exception(
                    "Failed converting case {0} to ff. max diff: {1}".format(i, maxdiff))
        ffts.append(myfft)
    print("OK")

# test idft
def test_idft():
    print("Testing 1d IDFT...")
    for i,rf in enumerate(ffts):
        myidft = sol2.IDFT(rf)
        npidft = np.fft.ifft(rf)
        if myidft.dtype != np.complex128:
            raise Exception("Failed in IDFT - returned wrong type")
        npidft = np.real(npidft)
        if not np.all(myidft == npidft):
            maxdiff = np.absolute(myidft - npidft).max()
            if maxdiff > EPSILON:
                raise Exception(
                    "Failed converting case {0} to ff. max diff: {1}".format(i, maxdiff))
    print("OK")

#test dft2
def test_dft2():
    print("Testing 2d DFT...")
    ffts[:] = []
    for impath in images_grey:
        im = sol1.read_image(impath, 1)
        npfft = np.fft.fft2(im)
        myfft = sol2.DFT2(im)
        if myfft.dtype != np.complex128:
            raise Exception("Failed in DFT2 - returned wrong type")
        if not np.all(npfft == myfft):
            avgdiff = np.absolute(npfft - myfft).mean()
            if avgdiff > EPSILON:
                raise Exception("Failed converting image in {0} to fft. mean diff: {1}".format(impath, avgdiff))
        ffts.append(myfft)
    print("OK")

#test idft2
def test_idft2():
    print("Testing 2d IDFT...")
    for i,rf in enumerate(ffts):
        myidft = sol2.IDFT2(rf)
        npifft = np.fft.ifft2(rf)
        if myidft.dtype != np.complex128:
            raise Exception("Failed in IDFT2 - returned wrong type")
        if not np.all(myidft == npifft):
            avgdiff = np.absolute(npifft - myidft).mean()
            if avgdiff > EPSILON:
                plt.figure()
                plt.suptitle("Failed image: " + images_grey[i])
                plt.subplot(1,3,1)
                plt.title('Expected')
                plt.axis('off')
                plt.imshow(sol1.read_image(images_grey[i], 1), cmap=plt.cm.gray)
                plt.subplot(1, 3, 2)
                plt.title('NP ifft2 for your DFT2')
                plt.axis('off')
                plt.imshow(npifft, cmap=plt.cm.gray, )
                plt.subplot(1, 3, 3)
                plt.title('Your IDFT2 for your DFT2')
                plt.axis('off')
                plt.imshow(myidft, cmap=plt.cm.gray, )
                plt.show()
                pylab.show(block=True)
                raise Exception(
                    "Failed converting case {0} to iff. avg diff: {1}".format(images_grey[i], avgdiff))
    print("OK")


def test_conv_der():
    print("Testing conv_der...")
    plt.figure()
    plt.suptitle('Conv der')
    for i,impath in enumerate(images_grey):
        im = sol1.read_image(impath, 1)
        mag_im = sol2.conv_der(im)
        plt.subplot(1, len(images_grey), i+1)
        plt.axis('off')
        plt.imshow(mag_im, cmap=plt.cm.gray)
    plt.show()
    print("OK")


#test fourier_der magnitude
def test_fourier_der():
    print("Testing fourier_der...")
    plt.figure()
    plt.suptitle('Fourier der')
    for i,impath in enumerate(images_grey):
        im = sol1.read_image(impath, 1)
        mag_im = sol2.fourier_der(im)
        plt.subplot(1, len(images_grey), i+1)
        plt.axis('off')
        plt.imshow(mag_im, cmap=plt.cm.gray)
    plt.show()
    print("OK")

def test_gauss_ker():
    print("Testing gaussian kernel generation..")
    for sz in range(1,151,2):
        ker = sol2.get_gaus_ker(sz)
        if ker.dtype != np.float32 and ker.dtype != np.float64:
            raise Exception("Kernel of size {0} is of the wrong type".format(sz))
        r,c = ker.shape
        if r != c or r != sz:
            raise Exception("Kernel of size {0} is of the wrong size {1}".format(sz, ker.shape))
        if ker.sum().astype(np.float32) != 1.0:
            raise Exception("Kernel of size {0} has a wrong sum {1}".format(sz, ker.sum()))
        if ker.min() <= 0:
            raise Exception("Kernel of size {0} has a wrong min {1}".format(sz, ker.min()))
        for i in range(sz//2):
            if not np.all(ker[i,:] == ker[-i-1,:]) or not np.all(ker[:,i] == ker[:,-i-1]):
                raise Exception("Kernel of size {0} has a non symetric matrix".format(sz))
    print("OK")

#test blur_spatial
def test_blur_spatial():
    print("Testing blur_spatial...")
    for ker_size in [5,7,9,15,21]:
        plt.figure()
        plt.suptitle("Spatial blur - Size: {0}".format(ker_size))
        for i, impath in enumerate(images_grey):
            print("Blurring img {0}/{1} ({2}) with kernel size {3}".format(i+1, len(images_grey), impath, ker_size))
            im = sol1.read_image(impath, 1)
            blur_im = sol2.blur_spatial(im, ker_size)
            plt.subplot(len(images_grey), 2, 2*i + 1)
            plt.axis('off')
            plt.imshow(im, cmap=plt.cm.gray)
            plt.title("Expected")

            plt.subplot(len(images_grey), 2, 2*i + 2)
            plt.axis('off')
            plt.imshow(blur_im, cmap=plt.cm.gray)
            plt.title("Blurred")
        plt.show()
    print("OK")

#test blur_fourier
def test_blur_fourier():
    print("Testing blur_spatial...")
    for ker_size in [7,9,15,21]:
        plt.figure()
        plt.suptitle("Fourier blur - Size: {0}".format(ker_size))
        for i, impath in enumerate(images_grey):
            print("Blurring img {0}/{1} ({2}) with kernel size {3}".format(i+1, len(images_grey), impath, ker_size))
            im = sol1.read_image(impath, 1)
            blur_im = sol2.blur_fourier(im, ker_size)
            plt.subplot(len(images_grey), 2, 2*i + 1)
            plt.axis('off')
            plt.imshow(im, cmap=plt.cm.gray)
            plt.title("Expected")

            plt.subplot(len(images_grey), 2, 2*i + 2)
            plt.axis('off')
            plt.imshow(blur_im, cmap=plt.cm.gray)
            plt.title("Blurred")
        plt.show()
    print("OK")

def compare_blurs():
    print("Testing compare_blurs...")
    for ker_size in [5, 15, 21, 31]:
        for i, impath in enumerate(images_grey):
            print("Blurring img {0}/{1} ({2}) with kernel size {3}".format(i+1, len(images_grey), impath, ker_size))
            im = sol1.read_image(impath, 1)
            blur_im_f = sol2.blur_fourier(im, ker_size)
            blur_im_s = sol2.blur_spatial(im, ker_size)
            avg_diff = np.mean(np.abs(blur_im_f-blur_im_s))
            if avg_diff > 1e-9:
                raise Exception("Failed comparison!")
    print("OK")

# TODO - test fourier against conv
def run_all_tests():
    print("Testing only grey. starting")
    test_blur_fourier()
    try:
        for test in [test_dft, test_idft, test_dft2, test_idft2, test_conv_der, test_fourier_der,
                     test_gauss_ker, test_blur_spatial, test_blur_fourier, compare_blurs]:
            test()
    except Exception as e:
        print("Tests failed. error: {0}".format(e))
        exit(-1)
    pylab.show(block=True)
    print("All tests passed!")

run_all_tests()
