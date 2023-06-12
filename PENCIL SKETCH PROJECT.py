#!/usr/bin/env python
# coding: utf-8

# # task:2 (BEGINNER LEVEL TASK): Image to Pencil Sketch with Python
# 
# NAME: AMULYA LOKANALLI

#  IMPORT LIBRARIES

# In[9]:


import cv2
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from PIL import Image
import matplotlib.pyplot as plt


# Read Image

# In[23]:


image = Image.open('C:\\Users\\amuly\\downloads\\WhatsApp Image 2023-06-12 at 11.04.39 AM.jpeg')


# In[24]:


image = plt.imread('C:\\Users\\amuly\\downloads\\WhatsApp Image 2023-06-12 at 11.04.39 AM.jpeg')


#  Display image and read the image in RGB format

# In[25]:


plt.imshow(image)
plt.axis('off')  # Optional: Disable axis labels
plt.show()


# Converting original image to greyscale image

# In[26]:


grey_scale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(grey_scale_image)


#  Inverting Image

# In[27]:


inverted_image = cv2.bitwise_not(grey_scale_image)
plt.imshow(inverted_image)


#  Blur image by using the Gaussian Function

# In[28]:


blurred_image = cv2.GaussianBlur(inverted_image, (111,111), 0)
plt.imshow(blurred_image)


# In[29]:


inverted_blur_image = cv2.bitwise_not(blurred_image)
plt.imshow(inverted_blur_image)


#  Invert the blur image for pencil skech

# In[30]:


sketch_image = cv2.divide(grey_scale_image, inverted_blur_image, scale=256.0)
plt.imshow(sketch_image)


# In[31]:


cv2.imwrite("sketch.png", sketch_image)


# #Final Image

# In[32]:


plt.figure(figsize = (8,8))
image3 = cv2.cvtColor(sketch_image, cv2.COLOR_RGB2BGR)
plt.imshow(image3)
plt.axis(False)
plt.show()

