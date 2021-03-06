import numpy as np
from keras.layers import Input, Convolution2D, Activation, merge
from keras.models import Model
from keras.optimizers import Adam
import sol5_utils
from scipy.ndimage.filters import convolve
from math import pi as PI

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
    cached_ims = {filename: None for filename in filenames}

    while True:
        # note that source_batch holds corrupted images! and target batch is the original (clean images)
        source_batch, target_batch = np.zeros((batch_size, dims, *crop_size)), np.zeros((batch_size, dims, *crop_size))
        for i in iters:
            # choose image and load it
            im_path = filenames[np.random.randint(0, nums_ims)]
            if cached_ims[im_path] is None:  # if not cached already, read and cache
                cached_ims[im_path] = read_image(im_path, REP_GREY)
            im = cached_ims[im_path]

            # randomly corrupt the image
            im_cor = corruption_func(im)

            # randomly choose patch to crop, and slice out the patch to the result
            r, c = im.shape
            y, x = np.random.randint(0, r - crop_size[0]), np.random.randint(0, c - crop_size[1])
            target_batch[i, 0, :, :] = im[y:y + crop_size[0], x:x + crop_size[1]] - 0.5
            source_batch[i, 0, :, :] = im_cor[y:y + crop_size[0], x:x + crop_size[1]] - 0.5

        yield source_batch.astype(np.float32), target_batch.astype(np.float32)


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

    :param height: height of each input-patch
    :param width: width of each input-patch
    :param num_channels: number of channel of each input-patch (1 for greyscale, 3 for rgb)
    :return: An untrained keras model.
    """
    NUM_RES_BLOCKS = 5
    input_a = Input(shape=(1, height, width))
    conv_b = Convolution2D(num_channels, 3, 3, border_mode='same')(input_a)
    relu_c = Activation('relu')(conv_b)
    curr_output_tensor = resblock(relu_c, num_channels)
    for res_block in range(NUM_RES_BLOCKS-1):
        curr_output_tensor = resblock(curr_output_tensor, num_channels)
    add_d = merge([curr_output_tensor, relu_c], mode='sum')
    conv_e = Convolution2D(1, 3, 3, border_mode='same')(add_d)
    return Model(input=input_a, output=conv_e)


def train_model(model, images, corruption_func, batch_size, samples_per_epoch, num_epochs, num_valid_samples):
    """
    The above function divides the images into a training set and validation set, using an 80-20 split,
    and generate from each set a dataset with the given batch size and corruption function (using the function
    from section 3). Then, the compile() method of the model using the mean square
    error loss and ADAM optimizer is invoked.

    The input arguments of the above function are:

    :param model:  a general neural network model for image restoration.
    :param images: a list of le paths pointing to image les. You should assume these paths are complete, and
                    should append anything to them.
    :param corruption_func: same as described in section 3.
    :param batch_size: the size of the batch of examples for each iteration of SGD.
    :param samples_per_epoch: The number of samples in each epoch (actual samples, not batches!).
    :param num_epochs: The number of epochs for which the optimization will run.
    :param num_valid_samples: The number of samples in the validation set to test on after every epoch.
                    Up to this point we have trained a neural network model on a given training set. Next, we move to
                    implementing the prediction step for restoring images.
    """
    # divide images to train and test
    crop_size = (model.input_shape[2], model.input_shape[3])
    num_images = len(images)
    shuffled_ims = np.asarray(images, dtype=object)[np.random.permutation(num_images)]
    train_set_size = int(np.ceil(0.8 * num_images))
    train_ims_gen = load_dataset(shuffled_ims[:train_set_size], batch_size, corruption_func, crop_size)
    val_ims_gen = load_dataset(shuffled_ims[train_set_size:], batch_size, corruption_func, crop_size)

    model.compile(optimizer=Adam(beta_2=0.9), loss='mean_squared_error')
    model.fit_generator(train_ims_gen, samples_per_epoch, num_epochs, validation_data=val_ims_gen,
                    nb_val_samples=num_valid_samples)


def restore_image(corrupted_image, base_model, num_channels):
    """
    Creates a new model that fits the size of the input image, and copy the weights from the
    given model to the adjusted model, before using the predict() method to restore the image.
    Finally, clips the results to the [0, 1] range.

    The above function has the following input arguments:

    :param corrupted_image: A grayscale image of shape (height, width) and with values in the [0, 1] range of type
                            float32. Assuming the size of the image is at least as large as the size of the image
                            patches during training.
    :param base_model: A neural network trained to restore small patches (the model described in section 4, after
                        being trained in section 5). The input and output of the network are images with values in the
                        [−0.5, 0.5] range
    :param num_channels: the number of channels used in the base model.
    :return restored_image - The restored image
    """
    model = build_nn_model(*corrupted_image.shape, num_channels)
    model.set_weights(base_model.get_weights())
    return np.clip(model.predict(
        corrupted_image[np.newaxis,np.newaxis, ...] - 0.5, batch_size=1)[0][0] + 0.5, 0, 1).astype(np.float32)


def add_gaussian_noise(image, min_sigma, max_sigma):
    """
    Randomly sample a value of sigma, uniformly distributed between min_sigma and
    max_sigma, followed by adding to every pixel of the input image a zero-mean gaussian random variable
    with standard deviation equal to sigma. Before returning the results, the values should be cliped to
    [0, 1].

    The input arguments to the function are:

    :param image: a grayscale image with values in the [0, 1] range of type float32.
    :param min_sigma: a non-negative scalar value representing the minimal variance of the gaussian distribution.
    :param max_sigma:  a non-negative scalar value larger than or equal to min_sigma, representing the maximal
                        variance of the gaussian distribution.
    :return: corrupted - the noisy image
    """
    sigma = np.random.uniform(min_sigma, max_sigma)
    noise = np.random.normal(0, sigma, image.size).reshape(image.shape)
    return np.clip(image + noise, 0, 1).astype(np.float32)


def learn_denoising_model(quick_mode=False):
    """
    The above method trains a network which expect patches of size 24×24,
    using 48 channels for all but the last layer. The corruption used is a gaussian noise with sigma
    in the range [0, 0.2].

    :param quick_mode: a single argument used solely for the presubmission phase
    :return: model, num_channels - The train models and the number fo channels used in it
    """
    images = sol5_utils.images_for_denoising()
    patch_size = (24, 24)
    num_channels = 48
    sigma_range = (0, 0.2)
    im_per_batch = 100 if not quick_mode else 10
    samples_per_epoch = 10000 if not quick_mode else 30
    total_epochs = 5 if not quick_mode else 2
    samples_for_validation = 1000 if not quick_mode else 30

    corrupt_im = lambda im: add_gaussian_noise(im, *sigma_range)

    model = build_nn_model(*patch_size, num_channels)
    train_model(model, images, corrupt_im, im_per_batch, samples_per_epoch, total_epochs, samples_for_validation)

    return model, num_channels


def add_motion_blur(image, kernel_size, angle):
    """
    The function add_motion_blur simulates motion blur on the given image using a square kernel
    of size kernel_size where the line (as described above) has the given angle in radians, measured
    relative to the positive horizontal axis, e.g. a horizontal line would have a zero angle, and for the
    figure above the angle would be 3π/4 (or 135◦ in degrees).
    :param image: a grayscale image with values in the [0, 1] range of type float32.
    :param kernel_size: an odd integer specifying the size of the kernel (even integers are ill-defined).
    :param angle: an angle in radians in the range [0, π).
    :return: corrupted - The corrupted image
    """
    ker = sol5_utils.motion_blur_kernel(kernel_size, angle)
    return convolve(image, ker, mode='reflect').astype(np.float32)


def  random_motion_blur(image, list_of_kernel_sizes):
    """
    The function random_motion_blur(image, list_of_kernel_sizes) samples an angle at uniform from the range
    [0, π), and chooses a kernel size at uniform from the list list_of_kernel_sizes, followed by applying
    the previous function with the given image and the randomly sampled parameters. The input arguments
    to the above two functions are:
    :param image: a grayscale image with values in the [0, 1] range of type float32.
    :param list_of_kernel_sizes: a list of odd integers.
    :return: corrupted - The corrupted image
    """
    angles_range = (0, PI)
    rnd_angle = np.random.uniform(*angles_range)
    rnd_size = np.random.choice(list_of_kernel_sizes)
    return add_motion_blur(image, rnd_size, rnd_angle)


def learn_deblurring_model(quick_mode=False):
    """
    The above method trains a network which expect patches of size 16×16,
    and have 32 channels in all layers except the last.

    :param quick_mode: a single argument used solely for the presubmission phase.
    :return: model, num_channels - The train models and the number fo channels used in it
    """
    images = sol5_utils.images_for_deblurring()
    patch_size = (16, 16)
    num_channels = 32
    kernel_sizes = [7]
    im_per_batch = 100 if not quick_mode else 10
    samples_per_epoch = 10000 if not quick_mode else 30
    total_epochs = 10 if not quick_mode else 2
    samples_for_validation = 1000 if not quick_mode else 30

    corrupt_im = lambda im: random_motion_blur(im, kernel_sizes)

    model = build_nn_model(*patch_size, num_channels)
    train_model(model, images, corrupt_im, im_per_batch, samples_per_epoch, total_epochs, samples_for_validation)

    return model, num_channels
