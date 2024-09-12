import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf, keras
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
import pickle


model = keras.saving.load_model('model.keras',)

with open('label_encode.pkl','rb') as file:
    label_encode = pickle.load(file)

with open('onehot_encode.pkl','rb') as file:
    onehot_encode = pickle.load(file)
    
with open('scale.pkl','rb') as file:
    scale = pickle.load(file)    
    
st.title('Customer Churn Prediction')
geography = st.selectbox('Geography',onehot_encode.categories_[0])
gender = st.selectbox('Gender',label_encode.classes_)
age = st.slider('Age',18,100)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',1,36)
num_of_products = st.slider('Number Of Products',1,8)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1])

input_data = [{
    'CreditScore':credit_score,
    'Gender':label_encode.transform([gender])[0],
    'Age':age,
    'Tenure':tenure,
    'Balance':balance,
    'NumOfProducts':num_of_products,
    'HasCrCard':has_cr_card,
    'IsActiveMember':is_active_member,
    'EstimatedSalary':estimated_salary,
}]

input_data = pd.DataFrame(input_data) # Converting to DataFrame
input_data[['CreditScore','Age','Balance','EstimatedSalary','Tenure']] = scale.transform(input_data[['CreditScore','Age','Balance','EstimatedSalary','Tenure']]) # Scaling the data
geo_encode = onehot_encode.transform([[geography]]) # One Hot encoding the data
input_data = pd.concat([input_data.reset_index(drop=True),geo_encode],axis=1) # Concatenating the whole data
print(input_data)
prediction = model.predict(input_data) # Predicting the model

st.write(prediction[0][0])
if prediction[0][0] > 0.5:
    st.write('The customer is likely to Churn')
else:
    st.write('The customer will Stay')