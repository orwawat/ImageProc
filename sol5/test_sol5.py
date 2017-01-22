from sol5 import *
import sol5_utils
import numpy as np
import matplotlib.pyplot as plt
import os

def show(im, imcor, imres):
    def s(img, num):
        plt.subplot(2, 2, num)
        plt.imshow(img, cmap=plt.cm.gray)
    plt.figure()
    s(im,1)
    s(imcor,2)
    s(imres,3)
    s(np.abs(im-imres),4)


def test_denoise():
    num_ims = 10
    model_path = "model.h5"
    print("Start denoising test")
    # if os.path.exists(model_path):

    model, num_channels = learn_denoising_model(False)
    model.save_weights(model_path)
    for i,im_path in enumerate(np.random.choice(sol5_utils.images_for_denoising(), size=num_ims)):
        print("Start denoising im num: {0}/{1}".format(i+1, num_ims))
        im = read_image(im_path, 1)
        corrupt_im = add_gaussian_noise(im, 0, 0.2)
        restored_im = restore_image(corrupt_im, model, num_channels)
        show(im, corrupt_im, restored_im)
    plt.show()
    print("Done denoising test")

def test_denoise_wo_train():
    num_ims = 10
    model_path = "model.h5"
    print("Start denoising w/o train - test")
    # if os.path.exists(model_path):
    patch_size = (24, 24)
    num_channels = 48
    model = build_nn_model(*patch_size, num_channels)
    model.load_weights(model_path)
    for i, im_path in enumerate(np.random.choice(sol5_utils.images_for_denoising(), size=num_ims)):
        print("Start denoising im num: {0}/{1}".format(i + 1, num_ims))
        im = read_image(im_path, 1)
        corrupt_im = add_gaussian_noise(im, 0, 0.2)
        restored_im = restore_image(corrupt_im, model, num_channels)
        show(im, corrupt_im, restored_im)
    plt.show()
    print("Done denoising test")


def test_deblur():
    num_ims = 10
    model_path = "model_blur.h5"
    print("Start debluring test")
    # if os.path.exists(model_path):

    model, num_channels = learn_deblurring_model(False)
    model.save_weights(model_path)
    for i, im_path in enumerate(np.random.choice(sol5_utils.images_for_deblurring(), size=num_ims)):
        print("Start debluring im num: {0}/{1}".format(i+1, num_ims))
        im = read_image(im_path, 1)
        kernel_sizes = [7]
        corrupt_im = random_motion_blur(im, kernel_sizes)
        restored_im = restore_image(corrupt_im, model, num_channels)
        show(im, corrupt_im, restored_im)
    plt.show()
    print("Done debluring test")

def test_deblur_wo_train():
    num_ims = 10
    model_path = "model_blur.h5"
    print("Start debluring w/o train - test")
    # if os.path.exists(model_path):
    patch_size = (16, 16)
    num_channels = 32
    kernel_sizes = [7]
    model = build_nn_model(*patch_size, num_channels)
    model.load_weights(model_path)
    for i, im_path in enumerate(np.random.choice(sol5_utils.images_for_deblurring(), size=num_ims)):
        print("Start debluring im num: {0}/{1}".format(i + 1, num_ims))
        im = read_image(im_path, 1)
        corrupt_im = random_motion_blur(im, kernel_sizes)
        restored_im = restore_image(corrupt_im, model, num_channels)
        show(im, corrupt_im, restored_im)
    plt.show()
    print("Done debluring test")

def main():
    for t in [test_deblur]:
        t()

if __name__ == '__main__':
    main()
