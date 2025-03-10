#!/usr/bin/env python
# coding: utf-8

# ### Assumptions in Multiple Linear regression
# 
# 1. Linearity : The relationship- between the predictors(x) and the response(y) is linear
# 2. Independence : Observations are independent of each other
# 3. Homoscedasticity: The residuals (Y-Y_hat)) exhibit constant variance at all levels of the predictor
# 4. Normal Distribution of Errors : The residuals of the model are normally distributed
# 5. No multicollinearity: The independent variables should not be too highly correlated with each other

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
import numpy as np


# In[3]:


cars = pd.read_csv("Cars.csv")
cars.head()


# In[4]:


# Rearrange the columns 
cars = pd.DataFrame(cars, columns=["HP","VOL","SP","WT","MPG"])
cars.head()


# #### Description of Columns 
# - MPG : Milege of the car(mile per Gallon)(This is Y-column to be predicted)
# - HP : Horse Power of the car (X1 column)
# - VOL : volume of the car(size)(X2 column)
# - SP : Top speed of the car (Miles per Hour)(X3 column)
# - WT : Weight of the car (Pounds)(X4 column)

# In[6]:


cars.info()


# In[7]:


cars.isnull().sum()


# #### Observations 
# - There are no missing Values
# - There are 81 observations
# - The datatypes of the columns are

# In[9]:


fig,(ax_box,ax_hist)= plt.subplots(2,sharex=True,gridspec_kw={"height_ratios": (.15,.85)})
sns.boxplot(data=cars,x='HP', ax=ax_box,orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars,x='HP', ax=ax_hist,bins=30,kde=True,stat='density')
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# ### observations
# -there are some extreme values observed in towars the right tail of SP and HP distributions
# - in VOL and WT columns, a few outliers are observed in both tails of thier distributions
# - the extrme values of cars data may have come from the specially designed nature of cars
# - as this is multi-dimensional data, the outliers with respect to spatial dimensions may have to be consdiered while building the regress model

# In[11]:


cars[cars.duplicated()]


# In[12]:


sns.set_style(style='darkgrid')
sns.pairplot(cars)


# In[13]:


cars.corr()


# ### observayions
# - btw vol and wt highest corr is 0.999203
# 
# - least is 0.077459 is the lowest btw vol and hp

# ### observations from correlation plots and coefficients
# - between x and y, all the x variables are showing moderate to high correlation strength, hishest being btw hp and mpg
# - therefore this dataset qualifies for building a multiple linear regression model to predict mpg
# - among x volumns(x1,x2,x3 and x4),some very high correlation strength are observed btw SP vs HP, VOl vs WT
# - the high correlation among x columns is not desirable as it might lead to multicollinearity problem

# In[16]:


model=smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit()


# In[17]:


model.summary()


# ### observations from model summary
# - the R-squared abd adjusted R-squared values are good and about 75% of variability in Y is explained by x columns
# - the probability value with respect to F-statistic is close to zero,indiacting that all or some of X columns by X columns
# - the p-values for vol and wt are higher than 5% indicating some interraction issue among themselves which need to be firther explored

# ### perfomance mertices for model1

# In[20]:


df1=pd.DataFrame()
df1['actual_y1']=cars['MPG']
df1.head()


# In[21]:


pred_y1=model.predict(cars.iloc[:,0:4])
df1['pred_y1']= pred_y1
df1.head()


# In[22]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df1["actual_y1"], df1["pred_y1"])
print("MSE :",mse)
print("RMSE :",np.sqrt(mse))


# ### Checking for multicollinearity among X -columns using VIF method

# In[24]:


cars.head()


# In[25]:


# Compute VIF values
rsq_hp = smf.ols('HP~WT+VOL+SP',data=cars).fit().rsquared
vif_hp = 1/(1-rsq_hp)

rsq_wt = smf.ols('WT~HP+VOL+SP',data=cars).fit().rsquared  
vif_wt = 1/(1-rsq_wt) 

rsq_vol = smf.ols('VOL~WT+SP+HP',data=cars).fit().rsquared  
vif_vol = 1/(1-rsq_vol) 

rsq_sp = smf.ols('SP~WT+VOL+HP',data=cars).fit().rsquared  
vif_sp = 1/(1-rsq_sp) 

# Storing vif values in a data frame
d1 = {'Variables':['Hp','WT','VOL','SP'],'VIF':[vif_hp,vif_wt,vif_vol,vif_sp]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame


# ### Observations
# - The ideal range of VIF values shall be between 0 to 10. However slightly higher values can be tolerated
# - As seen from the very high VIF values for VOL and WT , it is clear that they are prone to mutlicollinearity problem
# - Hence it is decided to drop one of the columns(either VOL or WT) to overcome the multicollinearity.
# - It is decided to drop WT and retain VOL column in further models

# In[27]:


cars1 = cars.drop("WT", axis = 1)
cars1.head()


# In[28]:


import statsmodels.formula.api as smf
model2 = smf.ols('MPG~VOL+SP+HP',data=cars1).fit()


# In[29]:


model2.summary()


# In[30]:


df2=pd.DataFrame()
df2['actual_y2']=cars['MPG']
df2.head()


# In[31]:


pred_y2=model.predict(cars.iloc[:,0:4])
df2["pred_y2"]= pred_y2
df2.head()


# In[32]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df2["actual_y2"], df2["pred_y2"])
print("MSE :",mse)
print("RMSE :",np.sqrt(mse))


# ### Observations froom model2 summary()
# - The adjusted R-squared value improved slightly to 0.76
# - All the p-values for model parameters are less than 5% hence they are significant
# - Therefore theHP,VOL,SP columns are finalized as the significant predictor for the MPG response variable
# - There is no improvement in MSE value

# In[57]:


cars1.shape


# #### Leverage(Hat Values):
# - Leverage values diagnose if a data point has an extreme value in terms of the independent variables.A point with high leverage has a great ability to influence the regression line. The threshold for consdering a point as having high leverage is typically set at3(k+1) in , where k is the number of predictors
# 
#   

# In[64]:


k = 3
n = 81
leverage_cutoff = 3*((k+1)/n)
leverage_cutoff


# In[72]:


from statsmodels.graphics.regressionplots import influence_plot 
influence_plot(model2,alpha=.05)

y=[i for i in range(-2,8)]
x=[leverage_cutoff for i in range(10)]
plt.plot(x,y,'r+')
plt.show()


# #### Observations
# - From the above plot,it is evident that data points 65,70,76,78,79,80 are the influencers.
# - As their H leverage values are higher and size is higher

# In[75]:


cars1[cars1.index.isin([65,70,76,78,79,80])]


# In[77]:


cars2=cars1.drop(cars1.index[[65,70,76,78,79,80]],axis=0).reset_index(drop=True)


# In[79]:


cars2


# ### Build model3 on Cars2 dataset

# In[82]:


model3 = smf.ols('MPG~VOL+SP+HP', data=cars2).fit()


# In[84]:


model3.summary()


# ### Performance Metrics for model3

# In[87]:


df3 = pd.DataFrame()
df3["actual_y3"] = cars2["MPG"]
df3.head()


# In[91]:


pred_y3 = model3.predict(cars2.iloc[:,0:3])
df3["pred_y3"] = pred_y3
df3.head()


# In[93]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df3["actual_y3"],df3["pred_y3"])
print("MSE :",mse)
print("RMSE :",np.sqrt(mse))


# #### Comparison of models
#                      
# 
# | Metric         | Model 1 | Model 2 | Model 3 |
# |----------------|---------|---------|---------|
# | R-squared      | 0.771   | 0.770   | 0.885   |
# | Adj. R-squared | 0.758   | 0.761   | 0.880   |
# | MSE            | 18.89   | 18.91   | 8.68    |
# | RMSE           | 4.34    | 4.34    | 2.94    |
# 
# 
# - **From the above comparison table it is observed that model3 is the best among all with superior performance metrics**

# plot the qq plot model residuals an also the y_hat vs residuals

# In[ ]:




