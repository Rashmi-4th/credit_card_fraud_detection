#!/usr/bin/env python
# coding: utf-8

# In[2]:


## Importing

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


import os
os.getcwd()
os.chdir(r'C:\Users\Rashmi\Desktop\Projects\credit_card_fraudlent')
os.getcwd()


# In[4]:


df= pd.read_csv('creditcard.csv')
df.head()


# In[5]:


df.tail()


# In[6]:


df.isnull().sum()


# In[7]:


df.describe


# In[8]:


df.info()


# In[9]:


fraud = df.loc[df['Class'] == 1]
normal = df.loc[df['Class'] == 0]


# In[11]:


fraud


# In[12]:


normal


# In[13]:


fraud.sum()


# In[14]:


len(fraud)


# In[15]:


len(normal)


# In[16]:


sns.scatterplot(x='Amount', y='Time',hue='Class',data= df)


# In[19]:


df1 = df.sample(frac=0.1, random_state=1)
print(df1.shape)
print(df1.describe)


# In[20]:


df.hist(figsize=(20,20))
plt.show()


# In[21]:


plt.plot(df)


# In[23]:


### Using logistic regression
from sklearn.model_selection import train_test_split
x=df.iloc[:,:-1]
y=df['Class']


# In[30]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=10)


# In[31]:


from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(x_train,y_train)


# In[32]:


y_pred = np.array(logreg.predict(x_test))
y=np.array(y_test)


# In[34]:


from sklearn.metrics import confusion_matrix, classification_report,accuracy_score


# In[35]:


print(confusion_matrix(y_test,y_pred))


# In[36]:


print(accuracy_score(y_test,y_pred))


# In[37]:


print(classification_report(y_test,y_pred))


# In[38]:


print(classification_report(y,y_pred))


# In[39]:


print(accuracy_score(y,y_pred))


# In[ ]:




