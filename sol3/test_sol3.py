import numpy as np
import sol1
import sol3
import matplotlib.pyplot as plt
from matplotlib import pylab
from scipy.ndimage.filters import gaussian_filter
import os

EPSILON = 1e-7

image_pth = r'C:\Users\Maor\Documents\ImageProc\current\external'
#image_pth = r'/cs/usr/maor_i/safe/PycharmProjects/ImageProc/sol2/external'
images = [os.path.join(image_pth, 'jerusalem.jpg'),
            os.path.join(image_pth, 'Low_Contrast.jpg'),
            os.path.join(image_pth, 'monkey.jpg')]
      #      os.path.join(image_pth, 'head-shot.jpg')]
images_grey = images # + [os.path.join(image_pth, 'head-shot_grey.jpg')]

def sample_valid_image(imgpth):
    im = sol1.read_image(imgpth, 1)
    valid_height = 2 ** int(np.log2(im.shape[0]))
    valid_width = 2 ** int(np.log2(im.shape[1]))
    return im[:valid_height, :valid_width]

def test_gaus_pyr():
    max_size = 7
    for kersize in [3,5,7]:
        for impth in images_grey:
            im = sample_valid_image(impth)
            pyr, filter = sol3.build_gaussian_pyramid(im, max_size, kersize)

            if len(pyr) > max_size:
                raise Exception("Pyramid is too long")
            if pyr[-1].shape[0] < 16 or pyr[-1].shape[1] < 16:
                raise Exception("Pyramid top is too short")
            for p in pyr:
                if p.dtype != np.float32:
                    raise Exception("Pyramid has wrong dtype")


def test_laplac_pyr():
    max_size = 7
    for kersize in [3, 5, 7]:
        for impth in images_grey:
            im = sample_valid_image(impth)
            pyr, filter = sol3.build_laplacian_pyramid(im, max_size, kersize)

            if len(pyr) > max_size:
                raise Exception("Pyramid is too long")
            if pyr[-1].shape[0] < 16 or pyr[-1].shape[1] < 16:
                raise Exception("Pyramid top is too short")
            for p in pyr:
                if p.dtype != np.float32:
                    raise Exception("Pyramid has wrong dtype")


def test_render_pyr():
    max_size = 7
    for kersize in [3]:
        for impth in images_grey:
            im = sample_valid_image(impth)
            plt.figure()
            plt.title('Image: {0}, KerSize: {1}'.format(impth, kersize))
            pyr, filter = sol3.build_gaussian_pyramid(im, max_size, kersize)
            plt.subplot(2,1,1)
            sol3.display_pyramid(pyr, len(pyr))
            plt.subplot(2,1,2)
            pyr, filter = sol3.build_laplacian_pyramid(im, max_size, kersize)
            sol3.display_pyramid(pyr, len(pyr))

            plt.show()
            pylab.show(block=True)


def test_lap2image():
    max_size = 7
    for kersize in [3, 5, 7]:
        for impth in images_grey:
            im = sample_valid_image(impth)
            pyr, filter = sol3.build_laplacian_pyramid(im, max_size, kersize)
            rec_im = sol3.laplacian_to_image(pyr, filter, [1]*len(pyr))

            diff_im = np.abs(rec_im - im)
            if diff_im.max() > 1E-12:
                raise Exception("Reconstructed image is too different from original")


def run_all_tests():
    print("Testing only grey. starting")
    try:
        for test in [test_lap2image, test_gaus_pyr, test_laplac_pyr, test_render_pyr]:
            test()
    except Exception as e:
        print("Tests failed. error: {0}".format(e))
        exit(-1)
    pylab.show(block=True)
    print("All tests passed!")

run_all_tests()
