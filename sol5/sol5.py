import numpy as np
from keras.layers import Input, Convolution2D, Activation, merge
from keras.models import Model

# --------------------- From sol1 ------------------------
from skimage.color import rgb2gray
from scipy.misc import imread

# Constants
REP_GREY = 1
REP_RGB = 2
MIN_INTENSITY = 0
MAX_INTENSITY = 255
RGB2YIQ_MAT = np.array([0.299, 0.587, 0.114, 0.596, -0.275, -0.321, 0.212, -0.523, 0.311],
                       dtype=np.float32).reshape(3, 3)
YIQ2RGB_MAT = np.array([1, 0.956, 0.621, 1, -0.272, -0.647, 1, -1.106, 1.703], dtype=np.float32).reshape(3, 3)


def read_image(filename, representation):
    """
    Reads a given image file and converts it into a given representation
        filename - string containing the image filename to read.
        representation - representation code, either 1 or 2 defining if the output should be either a
                        grayscale image (1) or an RGB image (2).
        return im as np.float32 in range [0,1]
    """
    im = imread(filename)
    if (representation == REP_GREY) & (im.ndim == 2):
        return im.astype(np.float32) / MAX_INTENSITY
    elif (representation == REP_GREY) & (im.ndim == 3):
        return rgb2gray(im).astype(np.float32)
    elif representation == REP_RGB:  # assuming we are not asked to convert grey to rgb
        return im.astype(np.float32) / MAX_INTENSITY
    else:
        raise Exception('Unsupported representation: {0}'.format(representation))

# -----------------------------------------------------------


def load_dataset(filenames, batch_size, corruption_func, crop_size):
    """

    :param filenames: A list of clean images.
    :param batch_size: The size of the batch of images for each iteration of Stochastic Gradient Descent.
    :param corruption_func: A function receiving a numpy's array representation of an image as a single argument,
                            and returns a randomly corrupted version of the input image.
    :param crop_size: A tuple (height, width) specifying the crop size of the patches to extract.
    :return: data_generator - a Python's generator object which outputs random tuples of the form
                (source_batch, target_batch), where each output variable is an array of shape (batch_size, 1,
                height, width), target_batch is made of clean images, and source_batch is their respective randomly
                corrupted version according to corruption_func(im).
    """
    iters = range(batch_size)
    dims = 1
    nums_ims = len(filenames)
    cached_ims = {filename: 0 for filename in filenames}

    while True:
        source_batch, target_batch = np.zeros((batch_size, dims, *crop_size))
        for i in iters:
            # choose image and load it
            im_path = filenames[np.random.randint(0, nums_ims, 1)]
            if cached_ims[im_path] == 0:  # if not cached already, read and cahce
                cached_ims[im_path] = read_image(im_path, REP_GREY)
            im = cached_ims[im_path]

            # randomly corrupt the image
            im_cor = corruption_func(im)

            # randomly choose patch to crop, and slice out the patch to the result
            r, c = im.shape
            y, x = np.random.randint(0, r-crop_size[0], 1), np.random.randint(0, c-crop_size[1], 1)
            source_batch[i, 0, :, :] = im[y:y+crop_size[0], x:x+crop_size[1]]
            target_batch[i, 0, :, :] = im_cor[y:y+crop_size[0], x:x+crop_size[1]]

        yield source_batch, target_batch


def resblock(input_tensor, num_channels):
    """
    This function takes as input a symbolic input tensor and the number of channels for each of its
    convolutional layers, and returns the symbolic output tensor of the layer configuration described above.
    The convolutional layers use \same" border mode, so as to not decrease the spatial dimension of the output tensor.
    :param input_tensor:
    :param num_channels:
    :return: output_tensor
    """
    conv_a = Convolution2D(num_channels, 3, 3, border_mode='same')(input_tensor)
    relu_b = Activation('relu')(conv_a)
    conv_c = Convolution2D(num_channels, 3, 3, border_mode='same')(relu_b)
    output_tensor = merge([input_tensor, conv_c], mode='sum')
    return output_tensor


def build_nn_model(height, width, num_channels):
    """
    The function returns an untrained Keras model (not yet compiled), with input dimension the shape of
    (1, height, width), and all convolutional layers (including residual blocks) with number of output channels equal
    to num_channels, except the very last convolutional layer which has a single output channel.

    :param height:
    :param width:
    :param num_channels:
    :return:
    """
    NUM_RES_BLOCKS = 6
    input_a = Input(shape=(1, height, width))
    conv_b = Convolution2D(num_channels, 3, 3, border_mode='same')(input_a)
    relu_c = Activation('relu')(conv_b)
    curr_output_tensor = relu_c
    for res_block in range(NUM_RES_BLOCKS):
        curr_output_tensor = resblock(curr_output_tensor, num_channels)
    add_d = merge([curr_output_tensor, relu_c], mode='sum')
    conv_e = Convolution2D(1, 3, 3, border_mode='same')(add_d)
    return Model(input=input_a, output=conv_e)




