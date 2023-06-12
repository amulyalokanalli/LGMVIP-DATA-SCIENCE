#!/usr/bin/env python
# coding: utf-8

# # task1:Iris flower classification ML project
# 
# 
# NAME: AMULYA LOKANALLI
# 
# importing necessary libraries

# In[41]:


import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# In[17]:


df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',names = ['sepal length in cm',
                                                                                                  'sepal width in cm',
                                                                                                    'petal length in cm',
                                                                                                    'petal width in cm',
                                                                                                    'class' ])


# In[18]:


df


#  Exploratory Data Analysis

# In[19]:


df.describe()


# In[20]:


df.info()


#  Here .info() function tells us that there are no presence of null values in dataset.So here no need of data cleaning.

# In[21]:


sns.pairplot(df,hue='class')
plt.show()


# In[22]:


sns.boxplot(data=df, x="sepal width in cm", y="class")
plt.show()


# In[23]:


sns.boxplot(data=df, x="sepal length in cm", y="class")
plt.show()


# In[24]:


sns.boxplot(data=df, x="petal length in cm", y="class")
plt.show()


# In[25]:


sns.boxplot(data=df, x="petal width in cm", y="class")
plt.show()


#  Model Training and Evaluation.
# 
# The dataset is divide into training and testing

# In[26]:


data = df.values
X = data[:,0:4]
Y = data[:,4]


# In[33]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4)


#  For regression and classification we used ML model called 'Support vector Classifier'

# In[37]:


svn = SVC()
svn.fit(X_train, y_train)
predictions = svn.predict(X_test)
# Calculate the accuracy
accuracy_score(y_test, predictions)


# In[40]:


print(classification_report(y_test, predictions))


# In[ ]:




