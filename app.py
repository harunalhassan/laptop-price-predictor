import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained model and dataset
model = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# Set the title of the app
st.title("Laptop Price Predictor")

# Feature selection
brand = st.selectbox('Brand', df['BrandName'].unique())
type = st.selectbox('Type of Laptop', df['TypeName'].unique())
ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
weight = st.number_input('Weight of the Laptop')
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
ips = st.selectbox('IPS', ['No', 'Yes'])
screen_size = st.number_input('Screen Size')
resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2560x1440', '2560x1600'])
cpu = st.selectbox('CPU', df['CpuBrand'].unique())
hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD (in GB)', [0, 128, 256, 512, 1024])
gpu = st.selectbox('GPU', df['GpuBrand'].unique())

# Data Preprocessing
if touchscreen == 'Yes':
    touchscreen = 1
else:
    touchscreen = 0

if ips == 'Yes':
    ips = 1
else:
    ips = 0

X_res = int(resolution.split('x')[0])
Y_res = int(resolution.split('x')[1])
ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size

# Create input dataframe
query = pd.DataFrame({
    'BrandName': [brand],
    'TypeName': [type],
    'Ram': [ram],
    'Weight': [weight],
    'Touchscreen': [touchscreen],
    'Ips': [ips],
    'Ppi': [ppi],
    'CpuBrand': [cpu],
    'Hdd': [hdd],
    'Ssd': [ssd],
    'GpuBrand': [gpu]
})

# Make Prediction
prediction = model.predict(query)

# Display Prediction
st.write(f"The predicted price of this configuration is: ₹{prediction[0]:.2f}")
