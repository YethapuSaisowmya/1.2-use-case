#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import streamlit as st


# In[5]:


import io
get_ipython().run_line_magic('cd', '"C:\\Users\\saisowmya\\Desktop\\tele churn app"')


# In[6]:


CC=pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")


# In[9]:


st.title("Predict telecom customer churn")
st.markdown("Model to  predict churn")

st.header('features')
col1,col2,col3,col4,col5=st.columns(5)
with col1:
    Gender=st.selectbox('gender',options=CC.gender.unique())
    SeniorCitizen=st.selectbox('SeniorCitizen',options=CC.SeniorCitizen.unique())
    Partner=st.selectbox('Partner',options=CC.Partner.unique())
    Dependents=st.selectbox('Dependents',options=CC.Dependents.unique())
with col2:
    tenure=st.text_input('tenure',"Type Here")
    PhoneService=st.selectbox('PhoneService',options=CC.PhoneService.unique())
    MultipleLines=st.selectbox('MultipleLines',options=CC.MultipleLines.unique())
    InternetService=st.selectbox('InternetService',options=CC.InternetService.unique())
with col3:
    OnlineSecurity=st.selectbox('OnlineSecurity',options=CC.OnlineSecurity.unique())
    OnlineBackup=st.selectbox('OnlineBackup',options=CC.OnlineBackup.unique())
    DeviceProtection=st.selectbox('DeviceProtection',options=CC.DeviceProtection.unique())
    TechSupport=st.selectbox('TechSupport',options=CC.TechSupport.unique())
with col4:
    StreamingTV= st.selectbox( 'StreamingTV',options=CC.StreamingTV.unique())
    StreamingMovies=st.selectbox('StreamingMovies', options=CC.StreamingMovies.unique())
    Contract=st.selectbox('Contract',options=CC.Contract.unique())
    PaperlessBilling=st.selectbox('PaperlessBilling',options=CC.PaperlessBilling.unique())
with col5:
    PaymentMethod= st.selectbox('PaymentMethod',options=CC.PaymentMethod.unique())
    MonthlyCharges=st.text_input('MonthlyCharges',"Type Here")
    TotalCharges=st.text_input('TotalCharges',"Type Here")
    


# In[10]:


if st.button('predict'):
    result=predict(np.array([[gender,SeniorCitizen,Partner,Dependents,
       tenure,PhoneService,MultipleLines,InternetService,
       OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,
       StreamingTV, StreamingMovies,Contract,PaperlessBilling,
       PaymentMethod, MonthlyCharges, TotalCharges]]))
    st.text(result[0])


# In[ ]:




