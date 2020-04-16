#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("C:\\Users\\Rakshit Pensalwar\\Mini Projects\\ML Mini Project\\wine.csv")
df.head(10)


# In[3]:


df.iloc[:,1:].describe()


# In[4]:


for c in df.columns[1:]:
    df.boxplot(c,by='Class',figsize=(7,4),fontsize=14)
    plt.title("{}\n".format(c),fontsize=16)
    plt.xlabel("Wine Class", fontsize=16)


# **Here we can see that some features classify the wine labels clearly.** 
# For example: Alcalinity, Total Phenols, or Flavonoids produce boxplots with well-separated medians,means and quartiles which are clearly indicative of wine classes.
# 
# Below is an example of class seperation using two variables

# In[5]:


plt.figure(figsize=(10,6))
plt.scatter(df['OD280/OD315 of diluted wines'],df['Flavanoids'],c=df['Class'],edgecolors='k',alpha=0.75,s=150)
plt.grid(True)
plt.title("Scatter plot of two features showing the \ncorrelation and class seperation",fontsize=15)
plt.xlabel("OD280/OD315 of diluted wines",fontsize=15)
plt.ylabel("Flavanoids",fontsize=15)
plt.show()


# #### Features independent? Plot co-variance matrix
# 
# It can be seen that there are some good amount of correlation between features i.e. they are not independent of each other, as assumed in Naive Bayes technique. But we will go ahead and apply the classifier to see its performance.

# In[6]:


def correlation_matrix(df):
    fig = plt.figure(figsize=(16,12))
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap='coolwarm')
    ax1.grid(True)
    plt.title('Wine data set features correlation\n',fontsize=15)
    labels=df.columns
    ax1.set_xticklabels(labels,fontsize=9)
    ax1.set_yticklabels(labels,fontsize=9)
    fig.colorbar(cax, ticks=[0.1*i for i in range(-11,11)])
    plt.show()

correlation_matrix(df)


# # Principal Component Analysis

# In[7]:


from sklearn.preprocessing import StandardScaler


# In[8]:


scaler = StandardScaler()


# In[9]:


X = df.drop('Class',axis=1)
y = df['Class']


# In[10]:


X = scaler.fit_transform(X)


# In[11]:


dfx = pd.DataFrame(data=X,columns=df.columns[1:])


# In[12]:


dfx.head(10)


# In[13]:


dfx.describe()


# In[14]:


from sklearn.decomposition import PCA


# In[15]:


pca = PCA(n_components=None)


# In[16]:


dfx_pca = pca.fit(dfx)


# #### Plot the explained variance ratio

# In[17]:


plt.figure(figsize=(10,6))
plt.scatter(x=[i+1 for i in range(len(dfx_pca.explained_variance_ratio_))],
            y=dfx_pca.explained_variance_ratio_,
           s=200, alpha=0.75,c='orange',edgecolor='k')
plt.grid(True)
plt.title("Explained variance ratio of the \nfitted principal component vector\n",fontsize=25)
plt.xlabel("Principal components",fontsize=15)
plt.xticks([i+1 for i in range(len(dfx_pca.explained_variance_ratio_))],fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel("Explained variance ratio",fontsize=15)
plt.show()


# **The above plot means that the $1^{st}$ principal component explains about 36% of the total variance in the data and the $2^{nd}$ component explians further 20%. Therefore, if we just consider first two components, they together explain 56% of the total variance.**

# In[18]:


dfx_trans = pca.transform(dfx)


# In[19]:


dfx_trans = pd.DataFrame(data=dfx_trans)
dfx_trans.head(10)


# #### Plot the first two columns of this transformed data set with the color set to original ground truth class label

# In[20]:


plt.figure(figsize=(10,6))
plt.scatter(dfx_trans[0],dfx_trans[1],c=df['Class'],edgecolors='k',alpha=0.75,s=150)
plt.grid(True)
plt.title("Class separation using first two principal components\n",fontsize=20)
plt.xlabel("Principal component-1",fontsize=15)
plt.ylabel("Principal component-2",fontsize=15)
plt.show()


# In[21]:


plt.figure(figsize=(10,6))
plt.scatter(dfx_trans[0],dfx_trans[3],c=df['Class'],edgecolors='k',alpha=0.75,s=150)
plt.grid(True)
plt.title("Class separation using first and fourth principal components\n",fontsize=20)
plt.xlabel("Principal component-1",fontsize=15)
plt.ylabel("Principal component-4",fontsize=15)
plt.show()


# In[ ]:




