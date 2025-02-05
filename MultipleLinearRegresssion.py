#!/usr/bin/env python
# coding: utf-8

# Assumptions in Multilinear Regression 
# 
# 1. Linearity: The relationship between the predictors and the response is linear
# 2. Independence: Observations are independent of each other.
# 3. Homoscedasticity: The residuals (Y - Y_hat)) exhibit constant variance at all levels of the predictor.
# 4. Normal Distribution of Erros : The residuals of the model are normally distributed.
# 5. No Multicollinearity: The independent variables should not be too highly correlated with each other.

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot


# In[3]:


cars = pd.read_csv("Cars.csv")
cars.head()


# In[4]:


cars = pd.DataFrame(cars, columns=["HP","VOL","SP","WT","MPG"])
cars.head()


# ### Description of columns
# - MPG: Milege of the car(Mile per Gallon)(This is Y-column to be predicted)
# - HP: Horse Power of the car(X1 column)
# - VOL: Volume of the car(size)(X2 column)
# - SP: Top speed of the car(Miles per Hour)(X3 Column)
# - WT: Weight of the car(Pounds)(X4 Column)

# In[6]:


cars.info()
cars.isnull().sum()


# In[14]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex = True, gridspec_kw={"height_ratios":(.15,.85)})

sns.boxplot(data=cars, x='HP', ax=ax_box, orient='h')
ax_box.set(xlabel='')

sns.histplot(data=cars, x='HP', ax=ax_hist , bins =30, kde=True , stat="density")
ax_hist.set(ylabel='Density')

plt.tight_layout()
plt.show()


# In[16]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex = True, gridspec_kw={"height_ratios":(.15,.85)})

sns.boxplot(data=cars, x='VOL', ax=ax_box, orient='h')
ax_box.set(xlabel='')

sns.histplot(data=cars, x='VOL', ax=ax_hist , bins =30, kde=True , stat="density")
ax_hist.set(ylabel='Density')

plt.tight_layout()
plt.show()


# In[18]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex = True, gridspec_kw={"height_ratios":(.15,.85)})

sns.boxplot(data=cars, x='SP', ax=ax_box, orient='h')
ax_box.set(xlabel='')

sns.histplot(data=cars, x='SP', ax=ax_hist , bins =30, kde=True , stat="density")
ax_hist.set(ylabel='Density')

plt.tight_layout()
plt.show()


# In[20]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex = True, gridspec_kw={"height_ratios":(.15,.85)})

sns.boxplot(data=cars, x='WT', ax=ax_box, orient='h')
ax_box.set(xlabel='')

sns.histplot(data=cars, x='WT', ax=ax_hist , bins =30, kde=True , stat="density")
ax_hist.set(ylabel='Density')

plt.tight_layout()
plt.show()


# ### Observations
# - There are some extreme values(outliers) observed in towards the right tail of SP and HP distributions
# - In VOL and WT columns , a few outliers are observed in both tails of their distributions.
# - The extreme values of cars data may have come from the specially designed nature of cars.
# - As this is multi-dimensional data, the outliers with respect to spatial dimensions may have to be considered while building the regression model 
# 

# ### Checking for dulpicated rows

# In[24]:


cars[cars.duplicated()]


# ### Pair Plots and Correlation Coefficents

# In[28]:


sns.set_style(style='darkgrid')
sns.pairplot(cars)


# In[30]:


cars.corr()


# ### Observations from correlation plots and Coefficients
# - Between x and y, all the x variables are showing moderate to high correlation strengths, highest being between HP and MPG
# - Therefore this dataset qualifies for building a multiple linear regression model to predict MPG
# - Among x columns (x1,x2,x3 and x4), some very high correlation strengths are observed between SP vs HP,VOL vs WT
# - The high coorelation among x columns is not desirable as it might lead to multicollineaity problem 

# ### Preparing a preliminary model considering all X columns 

# In[34]:


model = smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit()


# In[36]:


model.summary()


# In[ ]:




