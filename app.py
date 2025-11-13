import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# ------------------------------
# Load the trained model
# ------------------------------
model = tf.keras.models.load_model("model.keras", compile=False)


# ------------------------------
# Load the encoders and scaler
# ------------------------------
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# ------------------------------
# Streamlit app
# ------------------------------
st.title('🏦 Customer Churn Prediction App')

# User inputs
geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('👤 Gender', label_encoder_gender.classes_)
age = st.slider('🎂 Age', 18, 92, 30)
balance = st.number_input('💰 Balance', min_value=0.0)
credit_score = st.number_input('💳 Credit Score', min_value=0.0)
estimated_salary = st.number_input('💼 Estimated Salary', min_value=0.0)
tenure = st.slider('📅 Tenure (Years)', 0, 10, 3)
num_of_products = st.slider('🛍️ Number of Products', 1, 4, 1)
has_cr_card = st.selectbox('💳 Has Credit Card', [0, 1])
is_active_member = st.selectbox('✅ Is Active Member', [0, 1])

# ------------------------------
# Prepare input data
# ------------------------------
# Transform categorical features
try:
    gender_encoded = label_encoder_gender.transform([gender])[0]
except ValueError:
    st.error("⚠️ Invalid gender selected!")
    st.stop()

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)

# Create input DataFrame
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender_encoded],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# Combine one-hot encoded columns
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Ensure correct column order for scaler
if hasattr(scaler, 'feature_names_in_'):
    input_data = input_data[scaler.feature_names_in_]

# Scale input
input_data_scaled = scaler.transform(input_data)

# ------------------------------
# Predict churn
# ------------------------------
prediction = model.predict(input_data_scaled)
prediction_proba = float(prediction[0][0])

# ------------------------------
# Display result
# ------------------------------
st.write(f"### 🔢 Churn Probability: **{prediction_proba:.2f}**")

if prediction_proba > 0.5:
    st.error("⚠️ The customer is **likely to churn.**")
else:
    st.success("✅ The customer is **not likely to churn.**")
