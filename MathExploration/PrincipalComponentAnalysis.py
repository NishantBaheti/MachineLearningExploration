#!/usr/bin/env python
# coding: utf-8

# # Principal Component Analysis

# - Principal components analysis (PCA) is a popular approach for deriving a low-dimensional set of features from a large set of variables.
# - PCA is a technique for reducing the dimension of a n × p data matrix X. The first principal component direction of the data is that along which the observations vary the most
# - When faced with a large set of correlated variables, principal components allow us to summarize this set with a smaller number of representative variables that collectively explain most of the variability in the original set.

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 


# ## PCA with Scikit-Learn

# In[2]:


from sklearn.datasets import load_boston
from sklearn.decomposition import PCA


# In[3]:


print(PCA.__doc__)


# In[4]:


dataset = load_boston()


# In[5]:


dataset.keys()


# In[6]:


df = pd.DataFrame(dataset.data,columns=dataset.feature_names)
df["target"] = dataset.target
df.head()


# ### with 2D

# In[22]:


x_ax = "AGE"
y_ax = "LSTAT"
z_ax = "target"


# In[23]:


fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(projection="3d")
ax.set(
    xlabel=x_ax, 
    ylabel=y_ax,
    zlabel=z_ax
)
ax.scatter(xs=df[x_ax],ys=df[y_ax],zs=df[z_ax])
plt.show()


# In[24]:


values = PCA(n_components=1).fit(df[[x_ax,y_ax]]).transform(df[[x_ax,y_ax]])


# In[31]:


fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot()
ax.set(ylabel=z_ax)
ax.scatter(df[x_ax],df[z_ax],label=x_ax)
ax.scatter(df[y_ax],df[z_ax],label=y_ax)
ax.scatter(values,df[z_ax],label="pca")
ax.legend()
plt.show()


# ### with 3D

# In[11]:


x_ax = "AGE"
y_ax = "LSTAT"
z_ax = "target"
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(projection="3d")
ax.set(
    xlabel=x_ax, 
    ylabel=y_ax,
    zlabel=z_ax
)
ax.scatter(xs=df[x_ax],ys=df[y_ax],zs=df[z_ax])
plt.show()


# In[12]:


# values = df[["AGE","LSTAT","target"]].values


# # In[13]:


# pca_model = PCA(n_components=2).fit(values)


# # In[14]:


# pca_values = pca_model.transform(values)


# # In[15]:


# fig = plt.figure(figsize=(10,10))
# ax = fig.add_subplot(projection="3d")
# ax.scatter(xs=pca_values[:,0],ys=pca_values[:,1])
# plt.show()


# # In[ ]:




