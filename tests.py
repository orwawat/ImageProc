
import sol1 as sol1
import matplotlib.pyplot as plt
import numpy as np

images = [r"C:\Users\Maor\Documents\ImageProc\external\jerusalem.jpg",
          r"C:\Users\Maor\Documents\ImageProc\external\Low Contrast.jpg",
          r"C:\Users\Maor\Documents\ImageProc\external\monkey.jpg",
          'C:\\Users\\Maor\\Pictures\\head-shot.jpg']
images_grey = images + ['C:\\Users\\Maor\\Pictures\\head-shot_grey.jpg']
# grey read
# for i, impath in enumerate(images_grey):
#     plt.figure(1)
#     im = sol1.read_image(impath, sol1.REP_GREY)
#     if im.dtype != np.float32:
#         raise Exception("In image with path: '{0}' the type was wrong".format(impath))
#     if im.max() > 1 or im.min() < 0:
#         raise Exception("In image with path: '{0}' the range was wrong".format(impath))
#     plt.subplot(2, 3, i+1)
#     plt.imshow(im, cmap=plt.cm.gray)
# plt.show()

# color read
# for i, impath in enumerate(images):
#     plt.figure(2)
#     im = sol1.read_image(impath, sol1.REP_RGB)
#     if im.dtype != np.float32:
#         raise Exception("In image with path: '{0}' the type was wrong".format(impath))
#     if im.max() > 1 or im.min() < 0:
#         raise Exception("In image with path: '{0}' the range was wrong".format(impath))
#     plt.subplot(2, 3, i+1)
#     plt.imshow(im)
# plt.show()

# imdisplay - grey
# for impath in images_grey:
#     im = sol1.imdisplay(impath, sol1.REP_GREY)

# # imdisplay - color
# for impath in images:
#     im = sol1.imdisplay(impath, sol1.REP_RGB)

# tests histograms grey
# plt.figure()
# for i, impath in enumerate(images_grey):
#     plt.subplot(5, 6, 1+(i * 6))
#     im = sol1.read_image(impath, sol1.REP_GREY)
#     plt.imshow(im, cmap=plt.cm.gray) # regular
#     plt.subplot(5, 6, 2+(i * 6))
#     im_eq, hist_orig, hist_eq = sol1.histogram_equalize(im)
#     plt.imshow(im_eq, cmap=plt.cm.gray) # eq
#     plt.subplot(5, 6, 3 + (i * 6))
#     plt.plot(hist_orig)
#     plt.subplot(5, 6, 4 + (i * 6))
#     plt.plot(hist_eq)
#     plt.subplot(5, 6, 5+(i * 6))
#     plt.plot(np.cumsum(hist_orig))
#     plt.subplot(5, 6, 6 + (i * 6))
#     plt.plot(np.cumsum(hist_eq))
# plt.show()


# tests histograms color
# plt.figure()
# for i, impath in enumerate(images):
#     plt.subplot(4, 6, 1+(i * 6))
#     im = sol1.read_image(impath, sol1.REP_RGB)
#     plt.imshow(im)  # regular
#     plt.subplot(4, 6, 2+(i * 6))
#     im_eq, hist_orig, hist_eq = sol1.histogram_equalize(im)
#     plt.imshow(im_eq) # eq
#     plt.subplot(4, 6, 3 + (i * 6))
#     plt.plot(hist_orig)
#     plt.subplot(4, 6, 4 + (i * 6))
#     plt.plot(hist_eq)
#     plt.subplot(4, 6, 5+(i * 6))
#     plt.plot(np.cumsum(hist_orig))
#     plt.subplot(4, 6, 6 + (i * 6))
#     plt.plot(np.cumsum(hist_eq))
# plt.show()


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
        plt.subplot(2,2,2)
        plt.imshow(im_q, cmap=plt.cm.gray)  # regular
        plt.subplot(2,1,2)
        plt.plot(error)
        plt.show()
print("Done!")
'''


# sampleim = np.array([[ [200,100,50], [100,50,50] ],
#                     [ [60,70,80], [0,0,255] ]]).astype(np.uint8)
# iminyiq= sol1.rgb2yiq(sampleim)
# plt.imshow(sampleim)
# print(sampleim)
# #print(iminyiq)
# impermuted = np.transpose(sampleim, (2, 0, 1))
# imreshaped = impermuted.reshape(3, -1)
# #print(imreshaped)
# converted = sol1.convert_rep(sampleim, np.eye(3))
# print(converted)

# im = sol1.read_image('C:\\Users\\Maor\\Pictures\\head-shot.jpg', sol1.REP_GREY)

 # im = (im * 255).round().astype(np.uint8)
 # hist_orig = np.histogram(im, bins=256)[0]
 # hist_cumsum = np.cumsum(hist_orig)
 # pixs_per_seg = im.size / 5
 # a=np.where(hist_cumsum <= pixs_per_seg)
 # a=a[0][-1]
 # im_q, err = sol1.quantize(im, 5, 10)
 # plt.plot(err)
 # plt.show()
 # plt.imshow(im_q, cmap=plt.cm.gray)







# plt.figure()
imc = sol1.read_image('C:\\Users\\Maor\\Pictures\\head-shot.jpg', sol1.REP_RGB)
error = [0] * 10
for q in range(10):
    print('Starting round: ', q)
    plt.subplot(3,4,q+1)
    im_quant, error[q] = sol1.quantize_rgb(imc, 2**(q+1), 10)
    plt.imshow(im_quant)
plt.subplot(3,4,11)
plt.imshow(imc)
plt.subplot(3,4,12)
plt.plot(np.arange(1,11), np.asarray(error))
plt.show()

# # for i in range(1,11):
# #     plt.subplot(5,2,i)
# #     plt.imshow(sol1.quantize(im, i, 10)[0], cmap=plt.cm.gray)
# # plt.show()
# plt.figure()
# for i in range(1,11):
#     plt.subplot(5,2,i)
#     plt.imshow(sol1.quantize(imc, i, 10)[0])
# plt.show()


im_eq, hior, hieq = sol1.histogram_equalize(im)
imc_eq, hior, hieq = sol1.histogram_equalize(imc)
plt.subplot(2,2,1)
plt.imshow(im, cmap=plt.cm.gray)
plt.subplot(2,2,2)
plt.imshow(im_eq, cmap=plt.cm.gray)
plt.subplot(2,2,3)
plt.imshow(imc)
plt.subplot(2,2,4)
plt.imshow(imc_eq)
plt.show()


'''