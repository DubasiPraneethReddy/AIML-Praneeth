#!/usr/bin/env python
# coding: utf-8

# In[52]:


n = []
def mean_value(n):
    for x in n:
        mean = sum(n)/len(n)
        return mean 


# In[61]:


n=[3,5,7]
mean_value(n)


# In[ ]:





# In[156]:


def mode_value(L):
    s = set(L)
    d = {}
    for x in s:
        d[x] = L.count(x)
    m = max(d.values())
    for k in d.keys():
        if d[k] == m:
            return k


# In[160]:


L= [12,23,23,23,43,43,43,43,54]
mode_value(L)


# In[ ]:




