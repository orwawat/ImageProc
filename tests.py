
import sol1 as sol1
import matplotlib.pyplot as plt
import numpy as np

images = [r"C:\Users\Maor\Documents\ImageProc\external\jerusalem.jpg",
          r"C:\Users\Maor\Documents\ImageProc\external\Low Contrast.jpg",
          r"C:\Users\Maor\Documents\ImageProc\external\monkey.jpg",
          'C:\\Users\\Maor\\Pictures\\head-shot.jpg']
images_grey = images + ['C:\\Users\\Maor\\Pictures\\head-shot_grey.jpg']
# grey read
for i, impath in enumerate(images_grey):
    plt.figure(1)
    im = sol1.read_image(impath, sol1.REP_GREY)
    if im.dtype != np.float32:
        raise Exception("In image with path: '{0}' the type was wrong".format(impath))
    if im.max() > 1 or im.min() < 0:
        raise Exception("In image with path: '{0}' the range was wrong".format(impath))
    plt.subplot(2, 3, i+1)
    plt.imshow(im, cmap=plt.cm.gray)
plt.show()

# color read
for i, impath in enumerate(images):
    plt.figure(2)
    im = sol1.read_image(impath, sol1.REP_RGB)
    if im.dtype != np.float32:
        raise Exception("In image with path: '{0}' the type was wrong".format(impath))
    if im.max() > 1 or im.min() < 0:
        raise Exception("In image with path: '{0}' the range was wrong".format(impath))
    plt.subplot(2, 3, i+1)
    plt.imshow(im)
plt.show()

# imdisplay - grey
for impath in images_grey:
    im = sol1.imdisplay(impath, sol1.REP_GREY)

# imdisplay - color
for impath in images:
    im = sol1.imdisplay(impath, sol1.REP_RGB)

# tests histograms grey
plt.figure()
for i, impath in enumerate(images_grey):
    plt.subplot(5, 6, 1+(i * 6))
    im = sol1.read_image(impath, sol1.REP_GREY)
    plt.imshow(im, cmap=plt.cm.gray) # regular
    plt.subplot(5, 6, 2+(i * 6))
    im_eq, hist_orig, hist_eq = sol1.histogram_equalize(im)
    if im_eq.dtype != np.float32:
        raise Exception("In image with path: '{0}' the type was wrong".format(impath))
    if im_eq.max() > 1 or im_eq.min() < 0:
        raise Exception("In image with path: '{0}' the range was wrong".format(impath))
    plt.imshow(im_eq, cmap=plt.cm.gray) # eq
    plt.subplot(5, 6, 3 + (i * 6))
    plt.plot(hist_orig)
    plt.subplot(5, 6, 4 + (i * 6))
    plt.plot(hist_eq)
    plt.subplot(5, 6, 5+(i * 6))
    plt.plot(np.cumsum(hist_orig))
    plt.subplot(5, 6, 6 + (i * 6))
    plt.plot(np.cumsum(hist_eq))
plt.show()


# tests histograms color
plt.figure()
for i, impath in enumerate(images):
    plt.subplot(4, 6, 1+(i * 6))
    im = sol1.read_image(impath, sol1.REP_RGB)
    plt.imshow(im)  # regular
    plt.subplot(4, 6, 2+(i * 6))
    im_eq, hist_orig, hist_eq = sol1.histogram_equalize(im)
    if im_eq.dtype != np.float32:
        raise Exception("In image with path: '{0}' the type was wrong".format(impath))
    if im_eq.max() > 1 or im_eq.min() < 0:
        raise Exception("In image with path: '{0}' the range was wrong".format(impath))
    plt.imshow(im_eq) # eq
    plt.subplot(4, 6, 3 + (i * 6))
    plt.plot(hist_orig)
    plt.subplot(4, 6, 4 + (i * 6))
    plt.plot(hist_eq)
    plt.subplot(4, 6, 5+(i * 6))
    plt.plot(np.cumsum(hist_orig))
    plt.subplot(4, 6, 6 + (i * 6))
    plt.plot(np.cumsum(hist_eq))
plt.show()


# test quantization grey
for impath in images_grey:
    quants = [1,2,3,4,5,10,20,30,50,80,100,125]
    im = sol1.read_image(impath, sol1.REP_GREY)
    for q in quants:
        fig = plt.figure()
        fig.suptitle('Q:{0}, im: {1}'.format(q, impath))
        plt.subplot(2,2,1)
        plt.imshow(im, cmap=plt.cm.gray)  # regular
        im_q, error = sol1.quantize(im, q, 25)
        if im_q.dtype != np.float32:
            raise Exception("In image with path: '{0}' the type was wrong".format(impath))
        plt.subplot(2,2,2)
        plt.imshow(im_q, cmap=plt.cm.gray)  # regular
        plt.subplot(2,1,2)
        plt.plot(error)
        plt.show()


# test quantization color
for impath in images:
    quants = [1,2,3,4,5,10,20,30,50,80,100,125]
    im = sol1.read_image(impath, sol1.REP_RGB)
    for q in quants:
        fig = plt.figure()
        fig.suptitle('Q:{0}, im: {1}'.format(q, impath))
        plt.subplot(2,2,1)
        plt.imshow(im)  # regular
        im_q, error = sol1.quantize(im, q, 25)
        if im_q.dtype != np.float32:
            raise Exception("In image with path: '{0}' the type was wrong".format(impath))
        plt.subplot(2,2,2)
        plt.imshow(im_q)  # regular
        plt.subplot(2,1,2)
        plt.plot(error)
        plt.show()

# test rgb quantization
for impath in images:
    quants = np.power(2, np.arange(1,11))
    im = sol1.read_image(impath, sol1.REP_RGB)
    error = [0] * len(quants)
    for i, q in enumerate(quants):
        fig = plt.figure()
        fig.suptitle('Q:{0}, im: {1}'.format(q, impath))
        plt.subplot(1,2,1)
        plt.imshow(im)  # regular
        plt.subplot(1,2,2)
        im_quant, error[i] = sol1.quantize_rgb(im, q, 10)
        if im_quant.dtype != np.float32:
            raise Exception("In image with path: '{0}' the type was wrong".format(impath))
        plt.imshow(im_quant)
    fig = plt.figure()
    fig.suptitle("Error for: " + impath)
    plt.plot(np.arange(1,11), np.asarray(error))
    plt.show()


print("Done!")