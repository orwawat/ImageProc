import numpy as np
import sol1
import sol3
import matplotlib.pyplot as plt
from matplotlib import pylab
from scipy.ndimage.filters import gaussian_filter
import os

EPSILON = 1e-7

image_pth = r'/cs/usr/maor_i/safe/PycharmProjects/ImageProc/sol2/external'
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
    print("Start testing gaussian pyramid")
    max_size = 7
    if sol3.get_filter_kernel(3).dtype != np.float32:
        raise Exception("Kernel type is wrong")

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
    print("Done ok")


def test_laplac_pyr():
    print("Start testing Laplasian pyramid")
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
    print("Done ok")

def test_render_pyr():
    print("Start testing rendering pyramids")
    max_size = 7
    for kersize in [3]:
        for impth in images_grey:
            im = sample_valid_image(impth)
            pyr, filter = sol3.build_gaussian_pyramid(im, max_size, kersize)
            sol3.display_pyramid(pyr, len(pyr))
            sol3.display_pyramid(pyr, len(pyr)-1)
            sol3.display_pyramid(pyr, len(pyr)-2)
            sol3.display_pyramid(pyr, len(pyr)-3)
            sol3.display_pyramid(pyr, len(pyr)-4)
            impyr = sol3.render_pyramid(pyr, len(pyr))

            pyr, filter = sol3.build_laplacian_pyramid(im, max_size, kersize)
            sol3.display_pyramid(pyr, len(pyr))
            sol3.display_pyramid(pyr, len(pyr) - 1)
            sol3.display_pyramid(pyr, len(pyr) - 2)
            sol3.display_pyramid(pyr, len(pyr) - 3)
            sol3.display_pyramid(pyr, len(pyr) - 4)
            impyr = sol3.render_pyramid(pyr, len(pyr))

            plt.show()
    plt.show()
    print("Done ok")

def test_lap2image():
    print("Start testing laplacian pyramid to image")
    max_size = 7
    for kersize in [3, 5, 7]:
        for impth in images_grey:
            im = sample_valid_image(impth)
            pyr, filter = sol3.build_laplacian_pyramid(im, max_size, kersize)
            rec_im = sol3.laplacian_to_image(pyr, filter, [4]+[1]*(len(pyr)-1))
            rec_im = sol3.laplacian_to_image(pyr, filter, [1]*(len(pyr)-1)+[4])
            rec_im = sol3.laplacian_to_image(pyr, filter, [1]*len(pyr))

            diff_im = np.abs(rec_im - im)
            print("Error: max - {0}, mean - {1}".format(diff_im.max(), diff_im.mean()))
            if diff_im.max() > 1E-7:
                raise Exception("Reconstructed image is too different from original. minDiff: {0}, meanDiff: {1}, maxDiff: {2}".format(diff_im.min(), diff_im.mean(), diff_im.max()))
    print("Done ok")


def test_examples():
    print("Start testing examples")
    sol3.blending_example1()
    sol3.blending_example2()
    print("Done")

def run_all_tests():
    print("Testing sol3. starting")
    try:
        for test in [test_gaus_pyr, test_laplac_pyr, test_render_pyr, test_lap2image, test_examples]:
            test()
    except Exception as e:
        print("Tests failed. error: {0}".format(e))
        exit(-1)
    pylab.show(block=True)
    print("All tests passed!")

run_all_tests()
