#!/usr/bin/env python
# coding: utf-8

# ## Step 1: Import the datasets and libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv('train-adview.csv')


# In[3]:


df


# In[4]:


df.head()


# In[5]:


df.dtypes


# In[6]:


df.tail()


# In[7]:


df.shape


# In[8]:


df.info()


# In[9]:


df.describe()


# ## Step 2 : Data Visualisation

# In[10]:


plt.hist(df["category"])
plt.show()


# In[11]:


plt.plot(df["adview"])
plt.show()


# In[12]:


df=df[df["adview"]<2000000]
df


# In[13]:


f,ax=plt.subplots(figsize=(10,0))
corr=df.corr()


# In[14]:


df_corr=sns.heatmap(corr,cmap=sns.color_palette("rainbow_r", as_cmap=True),square=True ,annot=True)
plt.show()


# ## Step 3 : Clean the dataset

# In[15]:


df.isnull()


# In[16]:


df.dropna()


# In[17]:


df.columns


# In[18]:


df['category']


# In[19]:


# adview is the target variable


# In[20]:


# mapping categories as a dictionary to real numbers
category={'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'H':8}


# In[21]:


df['category']=df['category'].map(category)


# In[22]:


df


# In[23]:


df['vidid']


# In[24]:


df['adview']


# In[25]:


df['category']


# In[26]:


df['views']


# In[27]:


df['likes']


# In[28]:


df['dislikes']


# In[29]:


df['comment']


# In[30]:


df['published']


# In[31]:


df['duration']


# In[32]:


# removing character "F" present in the dataset
df=df[df.views!='F']
df=df[df.likes!='F']
df=df[df.dislikes!='F']
df=df[df.comment!='F']


# In[33]:


df['views']


# ## Step 4 : Transform attributes into numerical values

# In[34]:


# convert values to integers for views,likes,comments,dislikes and adview
df['views']=pd.to_numeric(df['views'])
df['likes']=pd.to_numeric(df['likes'])
df['dislikes']=pd.to_numeric(df['dislikes'])
df['comment']=pd.to_numeric(df['comment'])
df['adview']=pd.to_numeric(df['adview'])


# In[35]:


col_vidid=df['vidid']


# In[36]:


col_vidid


# In[37]:


# encoding features
from sklearn.preprocessing import LabelEncoder
df['duration']=LabelEncoder().fit_transform(df['duration'])
df['vidid']=LabelEncoder().fit_transform(df['vidid'])
df['published']=LabelEncoder().fit_transform(df['published'])


# In[38]:


df.head()


# In[39]:


# libraries to convert time_in_sec for duration column
import datetime
import time


# In[40]:


def checki(x):
    y=x[2:]
    h=''
    m=''
    s=''
    mm=''
    P=['H','M','S']
    for i in y:
        if i not in P:
            mm+=i
        else:
            if(i=="H"):
                h=mm
                mm=''
            elif(i=="M"):
                m=mm
                mm=''
            else:
                s=mm
                mm=''
    if(h==''):
        h='00'
    if(m==''):
        m='00'
    if(s==''):
        s='00'
    bp=h+':'+m+':'+s
    return bp
    


# In[41]:


train=pd.read_csv("train-adview.csv")
mp = pd.read_csv("train-adview.csv")["duration"]
time = mp.apply(checki)
def func_sec(time_string):
    h,m,s = time_string.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)
time1=time.apply(func_sec)
df["duration"]=time1
df.head()


# ## Step 5 : Normalise and split the data into training and testing 

# In[42]:


# Split Data
Y_train = pd.DataFrame(data = df.iloc[:, 1].values, columns = ['target'])
df=df.drop(["adview"],axis=1)
df=df.drop(["vidid"],axis=1)
df.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df, Y_train, test_size=0.2, random_state=42)
X_train.shape
# Normalise Data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)
X_train.mean()


# ##  Step 6 : Use linear regression, support vector regressor, random forest
# 

# In[43]:


# Evaluation Metrics
from sklearn import metrics
def print_error(X_test, y_test, model_name):
    prediction = model_name.predict(X_test)
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, prediction))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, prediction))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))


# In[44]:


# Linear Regression
from sklearn import linear_model
linear_regression = linear_model.LinearRegression()
linear_regression.fit(X_train, y_train)
print_error(X_test,y_test, linear_regression)


# In[45]:


# Support Vector Regressor
from sklearn.svm import SVR
supportvector_regressor = SVR()
supportvector_regressor.fit(X_train,y_train)
print_error(X_test,y_test, linear_regression)


# ## Step 7 : Use Decision tree regressor and Random forest regressor

# In[46]:


# Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
decision_tree = DecisionTreeRegressor()
decision_tree.fit(X_train, y_train)
print_error(X_test,y_test, decision_tree)


# In[47]:


# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
n_estimators = 200
max_depth = 25
min_samples_split=15
min_samples_leaf=2
random_forest = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth, min_samples_split=min_samples_split)
random_forest.fit(X_train,y_train)
print_error(X_test,y_test, random_forest)


# In[ ]:




