
import sol1 as sol1
import matplotlib.pyplot as plt

im = sol1.read_image('C:\\Users\\Maor\\Pictures\\head-shot.jpg', sol1.REP_GREY)
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

# In[26]:

a= np.arange(10)
b=a*3
c = np.append(a, a)
np.interp(c,a,b)


# In[17]:

im = read_image('C:\\Users\\Maor\\Pictures\\head-shot.jpg', REP_RGB)
transmat = np.array([0.299, 0.587, 0.114, 0.596, -0.275, -0.321, 0.212, -0.523, 0.311]).reshape(3,3)
impermuted = np.transpose(im, (2, 0, 1))
print ('{0}, {1}'.format(im.shape, impermuted.shape))
imreshaped = impermuted.reshape(3,-1)
print (imreshaped.shape)


# In[18]:

print(im.dtype)
im2 = yiq2rgb(rgb2yiq(im))
print(im2.dtype)
dif = np.subtract(im, im2)
print (dif[50:60,50:60,0:1])


# In[18]:

imdisplay('C:\\Users\\Maor\\Pictures\\head-shot_grey.jpg', REP_GREY)


# In[16]:

im = read_image('C:\\Users\\Maor\\Pictures\\head-shot.jpg', REP_RGB)
print (im.dtype)
print (im.shape)
print(im.ndim)
print(np.min(im))
print(np.max(im))
plt.imshow(im)


# In[36]:

im = read_image('C:\\Users\\Maor\\Pictures\\head-shot.jpg', REP_GREY)
print (im.dtype)
print (im.shape)
print(im.ndim)
print(np.min(im))
print(np.max(im))
plt.figure()
plt.imshow(im,cmap=plt.cm.gray)
im = read_image('C:\\Users\\Maor\\Pictures\\head-shot_grey.jpg', REP_GREY)
print (im.dtype)
print (im.shape)
print(im.ndim)
print(np.min(im))
print(np.max(im))
plt.figure()
plt.imshow(im,cmap=plt.cm.gray)


# In[ ]:



