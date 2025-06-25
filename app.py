import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Load your model
model = load_model("lstm_model.h5")

# Label mapping
label_map = {0: "Normal", 1: "Warning", 2: "Critical"}

# Function to read HTML file
def load_html(file_path):
    with open(file_path, 'r') as file:
        return file.read()

# Load your HTML content
html_content = load_html(os.path.join("templates", "home.html"))

# Inject the HTML and CSS into Streamlit
st.markdown(html_content, unsafe_allow_html=True)

# Your input fields (can be kept separate from HTML form if needed)
time_step = st.number_input("Time Step")
pressure = st.number_input("Pressure")
temperature = st.number_input("Temperature")
vibration = st.number_input("Vibration")
corrosion = st.number_input("Corrosion Level")
flow_rate = st.number_input("Flow Rate")
salinity = st.number_input("Salinity")
age = st.number_input("Age of Pipe")
pressure_temp = st.number_input("Pressure-Temp Product")
corrosion_flow = st.number_input("Corrosion-Flow Ratio")

# Predict on button click
if st.button("Predict"):
    input_features = [
        time_step, pressure, temperature, vibration, corrosion,
        flow_rate, salinity, age, pressure_temp, corrosion_flow
    ]

    input_array = np.array(input_features).reshape(1, 1, -1)
    prediction = model.predict(input_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    result = label_map.get(predicted_class, "Unknown")

    st.success(f"Predicted Condition: **{result}**")
