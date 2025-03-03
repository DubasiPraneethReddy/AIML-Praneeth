#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


# In[4]:


iris = pd.read_csv("iris.csv")


# In[6]:


iris


# In[8]:


iris.info()


# In[10]:


counts = iris["variety"].value_counts()
plt.bar(counts.index, counts.values)


# In[12]:


iris.info()


# In[16]:


iris[iris.duplicated(keep=False)]


# In[21]:


iris = iris.drop_duplicates(keep = 'first')
iris


# In[23]:


iris[iris.duplicated]


# In[25]:


iris = iris.reset_index(drop = True)
iris


# In[27]:


labelencoder = LabelEncoder()
iris.iloc[:,-1] = labelencoder.fit_transform(iris.iloc[:,-1])
iris.head()


# In[29]:


iris.info()


# ### Observation 
# - The target column is still object type. It needs to be converted to numeric(int)

# In[32]:


iris['variety'] = pd.to_numeric(labelencoder.fit_transform(iris['variety']))
print(iris.info())


# In[34]:


X = iris.iloc[:,0:4]
Y = iris['variety']


# In[36]:


x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, random_state = 1)
x_train


# ### Building Decision Tree Classifier using Entropy Criteria 

# In[41]:


model = DecisionTreeClassifier(criterion = 'entropy',max_depth= None)
model.fit(x_train,y_train)


# In[43]:


plt.figure(dpi=1200)
tree.plot_tree(model);


# In[45]:


fn = ['sepal length (cm)','sepal width(cm)','petal length (cm)','petal width (cm)']
cn = ['setosa','versicolor','virginica']
plt.figure(dpi =1200)
tree.plot_tree(model,feature_names = fn, class_names=cn, filled = True);


# In[ ]:




