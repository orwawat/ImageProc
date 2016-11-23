
import sol1 as sol1
import matplotlib.pyplot as plt
import numpy as np

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
error = [0] * 11
for q in range(10):
    plt.subplot(3,4,q+1)
    im_quant, error[q] = sol1.quantize_rgb(imc, 2**(q+1), 10)
    plt.imshow(im_quant)
plt.subplot(3,4,11)
plt.imshow(imc)
plt.subplot(3,4,12)
plt.plot(np.arange(11), np.log(np.asarray(error)))
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