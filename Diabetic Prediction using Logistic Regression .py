#!/usr/bin/env python
# coding: utf-8

# In[1]:


## This dataset is originally from the national Institute of Diabetes and Digestive and Kindey
##Disease. The objective of the dataset is to diagnostically predict whether a patient has diabetes,
## on the section of these instance from a larger database. In particular , alll the patients here are females
## at lest 21 years old of pima Indian heritage.
## From the data set in the (.csv) file we can find serveral variables , some of them are independent 
## (serveral medical preddictor variable) and only the one target dependent variable (Outcome).


# ## Read data from csv

# In[45]:


import pandas as pd
import numpy as np


# In[4]:


data = pd.read_csv('diabetesdata.csv')
data.head()


# ## Seperate X data and Y data

# In[48]:


X=data.drop("Outcome" , axis=1)  #Croping x from dataset
Y=data["Outcome"]
X


# In[10]:


Y


# ## Traning and Testing Split

# In[58]:


from sklearn.model_selection import train_test_split

X_train,Y_train,X_test,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)


# In[66]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

sc= StandardScaler()
X_train = sc.fit_transform(X_train)
X_test= sc.fit_transform(X_test)
X_train.shape


# In[60]:


X_train=np.array(X_train)
X_test=np.array(X_test)
Y_train=np.array(Y_train)
Y_test=np.array(Y_test)


# In[61]:


X_train


# ## Logistic Regression Library

# In[67]:


from sklearn.linear_model import LogisticRegression 

Log = LogisticRegression(C=1 , max_iter=1000)

Log.fit(X_train,Y_train)


# ## Prediction of Output using Logistic regression object

# In[70]:


Y_p = Log.predict(X_test)
Y_p


# ## Calculating accuracy on Training data

# In[78]:


Score_on_Training= Log.score(X_train,Y_train)
print('Accuracy_On_Training : ', Score_on_Training*100)


# ## Calculating accuracy on Testing data

# In[80]:


Score_on_Testing= Log.score(X_test,Y_test)
print('Accuracy_On_Training : ', Score_on_Testing*100)


# ## Exactly which observations went wrong ??

# In[81]:


Y_p - Y_test


# ## What was the output of  sigmoid function for each observation in X_text

# In[83]:


Log.predict_proba(X_test)


# In[84]:


Log.predict_proba(X_test)[3]


# In[ ]:




