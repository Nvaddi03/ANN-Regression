import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle 

model  = tf.keras.models.load_model('regression_model.h5')
print('Model is loaded')

#load encoder and scalar
with open('label_encoder__reg_gender.pkl','rb') as file:
    lable_encoder_reg_gender = pickle.load(file)

with open('onehot_encoder_reg_geo.pkl','rb') as file:
    onehot_reg_encoder = pickle.load(file)

with open('scaler_reg.pkl','rb') as file:
    scaler_reg = pickle.load(file)

# streamlit app
st.title('Estimating Salary predictions')

# User input
geography = st.selectbox('Geography', onehot_reg_encoder.categories_[0])
gender = st.selectbox('Gender', lable_encoder_reg_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
#estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
exited = st.selectbox('Exited', [0,1])
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])


# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [lable_encoder_reg_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited': [exited]
})
geo_encoded = onehot_reg_encoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_reg_encoder.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler_reg.transform(input_data)


# Predict churn
prediction = model.predict(input_data_scaled)
prediction_sal = prediction[0][0]

st.write(f'Predicted Estimated Salary: ${prediction_sal:.2f}')

