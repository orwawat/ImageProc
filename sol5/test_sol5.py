from sol5 import *
import sol5_utils
import numpy as np
import matplotlib.pyplot as plt

def show(im, imcor, imres):
    s = lambda im, num: plt.subplot(2, 2, num); plt.imshow(im, cmap=plt.cm.gray)
    plt.figure()
    s(im,1)
    s(imcor,2)
    s(imres,3)
    s(np.abs(im-imres),4)


def test_denoise():
    model, num_channels = learn_denoising_model(True)
    for im_path in np.random.choice(sol5_utils.images_for_denoising(), size=10):
        im = read_image(im_path, 1)
        corrupt_im = add_gaussian_noise(im, 0, 0.2)
        restored_im = restore_image(corrupt_im, model, num_channels)
        show(im, corrupt_im, restored_im)
    plt.show()

def test_deblur():
    pass

def main():
    for t in [test_denoise, test_deblur]:
        t()

if __name__ == '__main__':
    main()
