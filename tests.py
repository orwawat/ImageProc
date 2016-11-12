
import sol1 as sol1
import matplotlib.pyplot as plt
import numpy as np

im = sol1.read_image('C:\\Users\\Maor\\Pictures\\head-shot.jpg', sol1.REP_GREY)

# im = (im * 255).round().astype(np.uint8)
# hist_orig = np.histogram(im, bins=256)[0]
# hist_cumsum = np.cumsum(hist_orig)
# pixs_per_seg = im.size / 5
# a=np.where(hist_cumsum <= pixs_per_seg)
# a=a[0][-1]
im_q, err = sol1.quantize(im, 5, 10)
plt.plot(err)
plt.show()
plt.imshow(im_q, cmap=plt.cm.gray)

imc = sol1.read_image('C:\\Users\\Maor\\Pictures\\head-shot.jpg', sol1.REP_RGB)
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