import matplotlib.pyplot as plt
import numpy as np

import current.sol1 as test_sol1
from extras import sol1 as sol1

images = [r"C:\Users\Maor\Documents\ImageProc\external\jerusalem.jpg",
          r"C:\Users\Maor\Documents\ImageProc\external\Low Contrast.jpg",
          r"C:\Users\Maor\Documents\ImageProc\external\monkey.jpg",
          'C:\\Users\\Maor\\Pictures\\head-shot.jpg']
images_grey = images + ['C:\\Users\\Maor\\Pictures\\head-shot_grey.jpg']
# grey read
# for i, impath in enumerate(images_grey):
#     plt.figure(1)
#     im = sol1.read_image(impath, 1)
#     if im.dtype != np.float32:
#         raise Exception("In image with path: '{0}' the type was wrong".format(impath))
#     if im.max() > 1 or im.min() < 0:
#         raise Exception("In image with path: '{0}' the range was wrong".format(impath))
#     plt.subplot(2, 3, i+1)
#     plt.imshow(im, cmap=plt.cm.gray)
# plt.show()
#
# # color read
# for i, impath in enumerate(images):
#     plt.figure(2)
#     im = sol1.read_image(impath, 2)
#     if im.dtype != np.float32:
#         raise Exception("In image with path: '{0}' the type was wrong".format(impath))
#     if im.max() > 1 or im.min() < 0:
#         raise Exception("In image with path: '{0}' the range was wrong".format(impath))
#     plt.subplot(2, 3, i+1)
#     plt.imshow(im)
# plt.show()
#
# # imdisplay - grey
# for impath in images_grey:
#     im = sol1.imdisplay(impath, 1)
#
# # imdisplay - color
# for impath in images:
#     im = sol1.imdisplay(impath, 2)

# tests histograms grey
#
for i, impath in enumerate(images_grey):
    im = sol1.read_image(impath, 1)
    im_eq, hist_orig, hist_eq = sol1.histogram_equalize(im)
    im_eq2, hist_orig2, hist_eq2 = test_sol1.histogram_equalize(im)

    if not np.all(hist_orig == hist_orig2):
        raise Exception("Not same hists!")
    if not np.all(hist_eq == hist_eq2):
        if np.sum(hist_eq-hist_eq2) != 0:
            raise Exception("Not same eq hists!")
    print("max diff is:{0}".format(np.max(np.abs(im_eq-im_eq2))*255))
    plt.imshow(np.abs(im_eq-im_eq2))
    if im_eq.dtype != np.float32:
        raise Exception("In image with path: '{0}' the type was wrong".format(impath))
    if im_eq.max() > 1 or im_eq.min() < 0:
        raise Exception("In image with path: '{0}' the range was wrong".format(impath))
print("Fine histo")


# tests histograms color
plt.figure()
for i, impath in enumerate(images):
    im_eq, hist_orig, hist_eq = sol1.histogram_equalize(im)
    im_eq2, hist_orig2, hist_eq2 = test_sol1.histogram_equalize(im)

    if not np.all(hist_orig == hist_orig2):
        raise Exception("Not same hists!")
    if not np.all(hist_eq == hist_eq2):
        if np.sum(hist_eq - hist_eq2) != 0:
            raise Exception("Not same eq hists!")
    print("max diff is:{0}".format(np.max(np.abs(im_eq - im_eq2)) * 255))
    if im_eq.dtype != np.float32:
        raise Exception("In image with path: '{0}' the type was wrong".format(impath))
    if im_eq.max() > 1 or im_eq.min() < 0:
        raise Exception("In image with path: '{0}' the range was wrong".format(impath))

print("color histo fine")


# test quantization grey
for impath in images_grey:
    quants = [1,2,3,4,5,10,20,30,50,80,100,125]
    im = sol1.read_image(impath, 1)
    for q in quants:
        im_q, error = sol1.quantize(im, q, 25)
        im_q2, error2= test_sol1.quantize(im, q, 25)

        if not np.all(error == error2):
            print("Different errors!!!")
            print(error)
            print(error2)

        print("max diff is:{0}".format(np.max(np.abs(im_q - im_q2)) * 255))
        if im_q.dtype != np.float32:
            raise Exception("In image with path: '{0}' the type was wrong".format(impath))

print("quant 1d is fine")

# test quantization color
for impath in images:
    quants = [1,2,3,4,5,10,20,30,50,80,100,125]
    im = sol1.read_image(impath, 2)
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
    im = sol1.read_image(impath, 2)
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