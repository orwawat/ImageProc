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


def test_super_res():
    num_ims = 10
    model_path = "model_super.h5"
    print("Start superres test")
    # if os.path.exists(model_path):

    model, num_channels = learn_super_resolution_model()
    model.save_weights(model_path)
    for i,im_path in enumerate(np.random.choice(sol5_utils.images_for_denoising(), size=num_ims)):
        print("Start superres im num: {0}/{1}".format(i+1, num_ims))
        im = read_image(im_path, 1)
        corrupt_im = expand(im, np.array([[0.25,0.5,0.25]]))
        restored_im = restore_image(corrupt_im, model, num_channels)
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(corrupt_im)
        plt.subplot(1,2,2)
        plt.imshow(restored_im)
    plt.show()
    print("Done superres test")

def test_suprres_wo_train():
    num_ims = 3
    model_path = "model_super.h5"
    print("Start superres w/o train - test")
    # if os.path.exists(model_path):
    patch_size = (32, 32)
    num_channels = 32
    model = build_nn_model(*patch_size, num_channels)
    model.load_weights(model_path)
    for i, im_path in enumerate(np.random.choice(sol5_utils.images_for_denoising(), size=num_ims)):
        print("Start superres im num: {0}/{1}".format(i + 1, num_ims))
        im = read_image(im_path, 1)
        imc = read_image(im_path, 2)
        corrupt_imc = np.zeros((imc.shape[0]*2, imc.shape[1]*2, 3))
        restored_imc = np.zeros(corrupt_imc.shape)
        corrupt_im = expand(im, np.array([[0.25, 0.5, 0.25]]))
        restored_im = restore_image(corrupt_im, model, num_channels)

        for c in range(3):
            corrupt_imc[:,:,c] = expand(imc[:,:,c], np.array([[0.25, 0.5, 0.25]]))
            restored_imc[:,:,c] = restore_image(corrupt_imc[:,:,c], model, num_channels)

        plt.figure()
        plt.subplot(2, 3, 1)
        plt.imshow(im, cmap=plt.cm.gray)
        plt.subplot(2, 3, 2)
        plt.imshow(corrupt_im, cmap=plt.cm.gray)
        plt.subplot(2, 3, 3)
        plt.imshow(restored_im, cmap=plt.cm.gray)
        plt.subplot(2, 3, 4)
        plt.imshow(imc)
        plt.subplot(2, 3, 5)
        plt.imshow(corrupt_imc)
        plt.subplot(2, 3, 6)
        plt.imshow(restored_imc)


    plt.show()
    print("Done debluring test")

def main():
    for t in [test_suprres_wo_train]:
        t()

if __name__ == '__main__':
    main()
