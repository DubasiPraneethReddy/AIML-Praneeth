#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import statsmodels.formula.api as smf 
import numpy as np 


# In[2]:


import pandas as pd 
cars = pd.read_csv("Toyoto_Corrola.csv")
cars.head()


# In[3]:


cars.info()


# In[4]:


cars1 = cars.drop(columns =["Model","Id"])
cars1 


# In[5]:


cars1.rename(columns = {'Age_08_04':'Age'}, inplace = True)
cars1 


# cars1.isna().sum()

# In[7]:


cars1[cars1.duplicated(keep = False)]


# #### Observations 
# - There are no null values
# - There are dulplicated rows with index 112, 113
# - The continous various columns :Price,Age,KM, HP and weight
# - The categorical columns are Doors, Cylinders and gears
# - The price columns is the predicted(y) variable 

# In[18]:


cars1.drop_duplicates(keep = 'first', inplace =True)
cars1.reset_index(drop =True)
cars1


# In[ ]:





# In[30]:


import seaborn as sns 
counts = cars1["Gears"].value_counts().reset_index()
print(counts)
sns.barplot(data = cars1, x = counts["Gears"], hue = counts["Gears"])


# In[ ]:





# In[ ]:




