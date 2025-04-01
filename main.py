import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

# Load the trained model
model = tf.keras.models.load_model("tf_bridge_model.h5")

# Load the saved scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Streamlit UI
st.title("Bridge Load Capacity Prediction")
st.write("Enter bridge details to predict the maximum load capacity (in tons).")

# User input fields
span_ft = st.number_input("Span (ft)", min_value=50, max_value=1000, value=250)
deck_width_ft = st.number_input("Deck Width (ft)", min_value=10, max_value=100, value=40)
age_years = st.number_input("Age (years)", min_value=0, max_value=200, value=20)
num_lanes = st.number_input("Number of Lanes", min_value=1, max_value=10, value=2)
condition_rating = st.slider("Condition Rating (1-5)", min_value=1, max_value=5, value=3)

# Material selection with encoding
material = st.selectbox("Material Type", ["Steel", "Concrete", "Composite"])
material_encoding = {"Steel": [0, 0], "Concrete": [1, 0], "Composite": [0, 1]}
material_features = material_encoding[material]

# Prepare input data for prediction
input_data = np.array([[span_ft, deck_width_ft, age_years, num_lanes, condition_rating] + material_features])
input_data[:, :5] = scaler.transform(input_data[:, :5])  # Scale numerical features

# Prediction
if st.button("Predict Load Capacity"):
    prediction = model.predict(input_data)
    max_load_tons = round(float(prediction[0][0]), 2)
    st.success(f"Predicted Maximum Load Capacity: {max_load_tons} tons")
