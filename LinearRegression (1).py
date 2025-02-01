#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# In[7]:


data1 = pd.read_csv("NewspaperData.csv")
print(data1)


# In[11]:


data1.info()


# In[13]:


data1.isnull().sum()


# In[15]:


data1.describe()


# In[17]:


plt.figure(figsize=(6,3))
plt.title("Box plot for Daily Sales")
plt.boxplot(data1["daily"],vert = False)
plt.show()


# In[21]:


sns.histplot(data1['daily'], kde = True, stat='density',)
plt.title("Daily Sales")
plt.show()


# In[25]:


plt.figure(figsize=(6,3))
plt.title("Box plot for Sunday Sales")
plt.boxplot(data1["sunday"],vert = False)
plt.show()


# In[34]:


sns.histplot(data1['sunday'], kde = True, stat='density',)
plt.title("Sunday Sales")
plt.show()


# ### Observations 
# - There are no missing values
# - The daily column values also appear to be right-skewed 
# - The sunday column values also appear to be right- skewed 
# - There are two outliers in both daily column and also in sunday column as observed from the boxplots

# ## Scatter plot and Correlation Strength

# In[37]:


x = data1["daily"]
y = data1["sunday"]
plt.scatter(data1["daily"], data1["sunday"])
plt.xlim(0, max(x) + 100)
plt.ylim(0, max(y) + 100)
plt.show()


# In[39]:


data1["daily"].corr(data1["sunday"])


# In[41]:


data1[["daily","sunday"]].corr()


# ### Obsevations 
# - The relationship between x(daily) and y(sunday) is seen to be linear as seen from scatter plot
# - The correlation is strong positive with Pearson's correlation coefficient od 0.958154

# ## Fit a linear Regression Model

# In[47]:


import statsmodels.formula.api as smf
model1 = smf.ols("sunday~daily",data = data1).fit()


# In[49]:


model1.summary()


# In[ ]:




