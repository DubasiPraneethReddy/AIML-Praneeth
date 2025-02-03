#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# In[2]:


data1 = pd.read_csv("NewspaperData.csv")
print(data1)


# In[3]:


data1.info()


# In[4]:


data1.isnull().sum()


# In[5]:


data1.describe()


# In[6]:


plt.figure(figsize=(6,3))
plt.title("Box plot for Daily Sales")
plt.boxplot(data1["daily"],vert = False)
plt.show()


# In[7]:


sns.histplot(data1['daily'], kde = True, stat='density',)
plt.title("Daily Sales")
plt.show()


# In[8]:


plt.figure(figsize=(6,3))
plt.title("Box plot for Sunday Sales")
plt.boxplot(data1["sunday"],vert = False)
plt.show()


# In[9]:


sns.histplot(data1['sunday'], kde = True, stat='density',)
plt.title("Sunday Sales")
plt.show()


# ### Observations 
# - There are no missing values
# - The daily column values also appear to be right-skewed 
# - The sunday column values also appear to be right- skewed 
# - There are two outliers in both daily column and also in sunday column as observed from the boxplots

# ## Scatter plot and Correlation Strength

# In[12]:


x = data1["daily"]
y = data1["sunday"]
plt.scatter(data1["daily"], data1["sunday"])
plt.xlim(0, max(x) + 100)
plt.ylim(0, max(y) + 100)
plt.show()


# In[13]:


data1["daily"].corr(data1["sunday"])


# In[14]:


data1[["daily","sunday"]].corr()


# ### Obsevations 
# - The relationship between x(daily) and y(sunday) is seen to be linear as seen from scatter plot
# - The correlation is strong positive with Pearson's correlation coefficient od 0.958154

# ## Fit a linear Regression Model

# In[17]:


import statsmodels.formula.api as smf
model1 = smf.ols("sunday~daily",data = data1).fit()


# In[18]:


model1.summary()


# In[36]:


x = data1["daily"].values
y = data1["sunday"].values
plt.scatter(x,y,color="m",marker = "o", s= 30)
b0 = 13.84
b1 = 1.33

y_hat = b0 + b1*x

plt.plot(x,y_hat, color = "g")

plt.xlabel('x')
plt.ylabel('y')
plt.show()


# In[38]:


sns.regplot(x="daily" , y="sunday", data=data1)
plt.xlim([0,1250])
plt.show()


# ## Predict for new data point

# In[41]:


newdata = pd.Series([200,300,1500])


# In[43]:


data_pred = pd.DataFrame(newdata,columns=["daily"])
data_pred


# In[45]:


model1.predict(data_pred)


# In[47]:


pred = model1.predict(data1["daily"])
pred


# In[49]:


data1["Y_hat"] = pred 
data1 


# In[51]:


data1["residuals"] = data1["sunday"]-data1["Y_hat"]
data1


# In[53]:


mse = np.mean((data1["daily"]-data1["Y_hat"])**2)
rmse = np.sqrt(mse)
print("MSE: ",mse)
print("RMSE: ",rmse)


# In[55]:


mae = np.mean(np.abs(data1["daily"]-data1["Y_hat"]))
mae


# In[ ]:




