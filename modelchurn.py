#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn


# In[2]:


import io
get_ipython().run_line_magic('cd', '"C:\\Users\\saisowmya\\Desktop\\tele churn app"')


# In[3]:


CC=pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")


# ### EDA

# In[5]:


CC.head()


# In[6]:


CC.tail()


# In[7]:


CC.info()


# In[8]:


CC.shape


# In[9]:


CC.columns


# In[10]:


CC.describe()


# In[11]:


#churn 
ax=CC.Churn.value_counts().plot(kind="bar",color="green")
for i in ax.containers:
    ax.bar_label(i,color="blue")
    


# Around 1800 customers left with in last month

# In[12]:


dx=pd.crosstab(CC.gender,CC.Churn).plot(kind="bar")
for i in dx.containers:
    dx.bar_label(i,color="red")


# We can see that equal number of male and female customers left from the telecom
# 

# In[13]:


pd.crosstab(CC.Churn,CC.SeniorCitizen)


# In[14]:


pd.crosstab(CC.Churn,CC.PhoneService)


# In[15]:


dx=pd.crosstab(CC.Churn,CC.InternetService).plot(kind='bar')
for i in dx.containers:
    dx.bar_label(i,color="red")


# From above bar plot we can say that the customers who had fiber optic internet servive left the telecom service
# 

# In[16]:


dx=pd.crosstab(CC.Churn,CC.MultipleLines).plot(kind='bar')
for i in dx.containers:
    dx.bar_label(i,color="red")


# In[17]:


pd.crosstab(CC.DeviceProtection,CC.Churn)


# customers who has not signed up for device protection service are leaving

# #### Hypothesis Testing

# test churn of differnet gender equal

# In[18]:


from scipy.stats import chi2_contingency


# In[19]:


chi2_contingency(pd.crosstab(CC['Churn'],CC['gender']))


# p- value is greater than 0.05,so we fail to reject null hypothesis

# #test null average tenure of churn 0/1 equal
# null - avg tenure of churn 0/1 is same
# alt - avg tenure of churn 0/1 are not same

# In[25]:


churnno=CC[CC['Churn']=='No']
churnyes=CC[CC['Churn']=='Yes']


# In[21]:


from scipy.stats import ttest_ind


# In[27]:


ttest_ind(churnno['tenure'],churnyes['tenure'],equal_var=False)
#pvalue=1.1954945472607151e-232  less than 0.05 --  reject Null


# ### Data Preprocessing

# ##### Feature Selection

# In[29]:


CC.drop('customerID',axis=1,inplace=True)


# ##### Checking for missing values

# In[30]:


CC.isna().sum()


# In[31]:


for i in CC.columns:
    print(i)
    print(CC[i].unique())
    print()


# Replacing NO internent service with No in CC columns

# In[32]:


CC.OnlineSecurity=CC.OnlineSecurity.replace('No internet service','No')
CC.OnlineBackup=CC.OnlineBackup.replace('No internet service','No')
CC.DeviceProtection=CC.DeviceProtection.replace('No internet service','No')
CC.TechSupport=CC.TechSupport.replace('No internet service','No')
CC.StreamingMovies=CC.StreamingMovies.replace('No internet service','No')
CC.StreamingTV=CC.StreamingTV.replace('No internet service','No')
CC.MultipleLines=CC.MultipleLines.replace('No phone service' ,'No')


#  TotalCharges should be an numeric column but it is given as object type.so converting it to float.

# In[33]:


CC['TotalCharges']=pd.to_numeric(CC['TotalCharges'],errors='coerce')


# In[35]:


CC.TotalCharges.isna().sum()


# In[36]:


#filling missing values with mean
CC.TotalCharges.fillna(CC.TotalCharges.mean(),inplace=True)


# ##### Error/outlier Detection

# In[37]:


seaborn.boxplot(CC)
plt.xticks(rotation= 90)
plt.show()


# There are no outliers in given dataset

# ##### Encoding

# In[39]:


cat_cols=['gender','Partner', 'Dependents','PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']


# In[40]:


cat_df=CC[['gender','Partner', 'Dependents','PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']]


# In[41]:


cat_dummy=pd.get_dummies(cat_df,columns=['gender','Partner', 'Dependents','PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'])


# In[42]:


CC.drop(['gender','Partner', 'Dependents','PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'],axis=1,inplace=True)


# In[43]:


CC=pd.concat((CC,cat_dummy),axis=1)


# In[44]:


from sklearn.preprocessing import LabelEncoder


# In[45]:


le = LabelEncoder()


# In[46]:


arr=le.fit_transform(CC['Churn'])


# In[47]:


le.classes_


# In[48]:


CC['Churn']=arr


# ##### Checking for imbalance

# In[49]:


ax=CC.Churn.value_counts().plot(kind='bar')
for i in ax.containers:
    ax.bar_label(i)


# ##### Data seperation

# In[50]:


X=CC.drop('Churn',axis=1)


# In[51]:


y=CC['Churn']


# In[52]:


from sklearn.preprocessing import StandardScaler


# In[53]:


cols_to_scale=['tenure', 'MonthlyCharges', 'TotalCharges']


# In[54]:


scaler=StandardScaler()


# In[55]:


scaler.fit(X[cols_to_scale])


# In[56]:


X[cols_to_scale]=scaler.transform(X[cols_to_scale])


# In[57]:


from imblearn.over_sampling import SMOTE


# In[58]:


smt=SMOTE()


# In[59]:


X_smt , y_smt = smt.fit_resample(X,y)


# In[60]:


ax=y_smt.value_counts().plot(kind = "bar")
for i in ax.containers:
    ax.bar_label(i)


# In[61]:


from sklearn.model_selection import train_test_split


# In[62]:


X_train, X_test, y_train, y_test = train_test_split(X_smt, y_smt,test_size=0.20,random_state=2)


# #### model building

# In[63]:


from sklearn.ensemble import RandomForestClassifier


# In[64]:


rf_model = RandomForestClassifier(n_estimators=300)


# In[65]:


rf_model.fit(X_train,y_train)


# In[66]:


rf_model.score(X_train,y_train)


# In[67]:


rfpredict=rf_model.predict(X_smt)


# In[69]:


from sklearn.metrics import classification_report


# In[70]:


print(classification_report(y_smt,rfpredict))


# In[72]:


# pickling the model 
import pickle 
pickle_out = open("classifier.pkl", "wb") 
pickle.dump(rf_model, pickle_out) 
pickle_out.close()


# In[73]:


import joblib


# In[74]:


joblib.dump(rf_model,'reg.sav')


# In[ ]:




