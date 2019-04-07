#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn import preprocessing
import quandl
quandl.ApiConfig.api_key = 'EuBXK6i-ViyNR-FCPLr_'
import pandas as pd


# In[3]:


online = False
def getStockData(path,stockName,Type):
    global online
    online = Type
    if online == True:
        return quandl.get('PSX/'+stockName.upper())
    if online == False:
        return pd.read_csv(path+stockName.upper()+'_01012010_06122018.csv')


# In[4]:


def feature_engineering(From,To,data):
    global online
    if online == True:
        return data.iloc[From:To,4],data.iloc[From:To,0],data.iloc[From:To,1],data.iloc[From:To,2],data.iloc[From:To,3]
    if online == False:
        #close,open,high,low,volume
        return data.iloc[From:To,5],data.iloc[From:To,2],data.iloc[From:To,3],data.iloc[From:To,4],data.iloc[From:To,6]


# In[5]:


def feature_arrange(From,To,data):
    return data.iloc[From:To,8],data.iloc[From:To,1],data.iloc[From:To,2],data.iloc[From:To,3],data.iloc[From:To,4],data.iloc[From:To,5],data.iloc[From:To,6],data.iloc[From:To,7],data.iloc[From:To,9]


# In[ ]:




