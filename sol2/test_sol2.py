import numpy as np
import sol1
import sol2
import matplotlib.pyplot as plt
from matplotlib import pylab

EPSILON = 1e-7

images = [r"C:\Users\Maor\Documents\ImageProc\external\jerusalem.jpg",
          r"C:\Users\Maor\Documents\ImageProc\external\Low Contrast.jpg",
          r"C:\Users\Maor\Documents\ImageProc\external\monkey.jpg",
          'C:\\Users\\Maor\\Pictures\\head-shot.jpg']
images_grey = images + ['C:\\Users\\Maor\\Pictures\\head-shot_grey.jpg']

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
        if myidft.dtype != np.float32:
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
    ffts = []
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
        if myidft.dtype != np.float32:
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
                plt.subplot(1, 3, 2)
                plt.title('Your IDFT2 for your DFT2')
                plt.axis('off')
                plt.imshow(myidft, cmap=plt.cm.gray, )
                plt.show()
                pylab.show(block=True)
                raise Exception(
                    "Failed converting case {0} to iff. avg diff: {1}".format(images_grey[i], avgdiff))
    print("OK")


#test conv_der magnitude
# TODO

#test fourier_der magnitude
# TODO

#test blur_spatial
# TODO

#test blur_fourier
# TODO
def run_all_tests():
    print("Testing only grey. starting")
    try:
        for test in [test_dft, test_idft, test_dft2, test_idft2]:
            test()
    except Exception as e:
        print("Tests failed. error: {0}".format(e))
        exit(-1)

run_all_tests()
