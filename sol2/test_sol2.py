import numpy as np
import sol1
import sol2
import matplotlib.pyplot as plt

EPSILON = 1e-10

images = [r"C:\Users\Maor\Documents\ImageProc\external\jerusalem.jpg",
          r"C:\Users\Maor\Documents\ImageProc\external\Low Contrast.jpg",
          r"C:\Users\Maor\Documents\ImageProc\external\monkey.jpg",
          'C:\\Users\\Maor\\Pictures\\head-shot.jpg']
images_grey = images + ['C:\\Users\\Maor\\Pictures\\head-shot_grey.jpg']

def stretch(im, mn=0, mx=1):
    return ((mx-mn)*(im-im.min()))/(im.max()- im.min())

print("Testing only grey!")

# test dft
# grey read
ffts = []
print("Testing 1d DFT...")
for i, r in enumerate([np.cos(np.arange(100)), np.sin(np.arange(100)), np.power(np.arange(100), 2), np.arange(100)]):
    im = stretch(r)
    npfft = np.fft.fft(im)
    myfft = sol2.DFT(im)
    if not np.all(npfft == myfft):
        maxdiff = np.absolute(npfft - myfft).max()
        if maxdiff > EPSILON:
            raise Exception(
                "Failed converting case {0} to ff. max diff: {1}".format(i, maxdiff))
    ffts.append(myfft)
print("OK")

# test idft
print("Testing 1d IDFT...")
for i,rf in enumerate(ffts):
    myidft = sol2.IDFT(rf)
    npidft = np.ifft(rf)
    if not np.all(myidft, npidft):
        raise Exception(
            "Failed converting case {0} to iff. max diff: {1}".format(i, np.abs(myidft - npidft).max()))
print("OK")

#test dft2
print("Testing 2d DFT...")
ffts = []
for impath in enumerate(images_grey):
    im = sol1.read_image(impath, 1)
    npfft = np.fft2(im)
    myfft = sol2.DFT2(im)
    if not np.all(npfft == myfft):
        raise Exception("Failed converting image in {0} to fft. max diff: {1}".format(impath, np.abs(npfft - myfft).max()))
    ffts.append(myfft)
print("OK")

#test idft2
print("Testing 2d IDFT...")
for i,rf in enumerate(ffts):
    myidft = sol2.IDFT2(rf)
    npidft = np.ifft2(rf)
    if not np.all(myidft, npidft):
        raise Exception(
            "Failed converting case {0} to iff. max diff: {1}".format(images_grey[i], np.abs(myidft - npidft).max()))
print("OK")


#test conv_der magnitude
# TODO

#test fourier_der magnitude
# TODO

#test blur_spatial
# TODO

#test blur_fourier
# TODO
