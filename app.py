import streamlit as st
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LeakyReLU
import base64
def set_background_image_local(image_path):
    with open(image_path, "rb") as file:
        data = file.read()
    base64_image = base64.b64encode(data).decode("utf-8")
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{base64_image}");
            background-size: cover;
            background-position: fit;
            background-repeat: repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background_image_local(r"12.png")
@st.cache_resource
def load_model_file():
    # Handle custom layers (if applicable)
    custom_objects = {"LeakyReLU": LeakyReLU}
    return load_model("model.h5", custom_objects=custom_objects)

# Load the model
model = load_model_file()

with open("std.pkl","rb") as f:
    scaler=pickle.load(f)

st.title("Predicting CNC Machine job Status")

Type_select=st.selectbox('Select Job Type',['Low','Medium','High'])
Type_map={'Low':1,'High':0,'Medium':2}
Type=Type_map.get(Type_select)
Air_Temperature = st.number_input('Enter Air Temperature', value=0.0, step=0.01, min_value=0.0)
Process_temperature = st.number_input('Enter Process temperature',value=0.0, step=0.01,min_value=0.0)
Rotational_speed_1= st.number_input('Enter Rotational speed',value=0.0, step=0.01,min_value=0.01)
Rotational_speed=1/(Rotational_speed_1**2)
Torque= st.number_input('Enter Torque',value=0.0, step=0.01,min_value=0.0)
Tool_wear=st.number_input('Enter Tool wear',value=0.0, step=0.01,min_value=0.0)

if st.button('Predict'):
    import numpy as np
    c1=np.array([[Type,Air_Temperature,Process_temperature,Rotational_speed,Torque,Tool_wear]])
    scaler.fit(c1)
    c2=scaler.transform(c1)
    predicted_output = model.predict(c2)
    predicted_class = np.argmax(predicted_output, axis=1)
    st.write("Job Status:", predicted_class)
    status={1:"No Failure",0:"Heat Dissipation Failure",3:"Power Failure",2:"Overstrain Failure",5:"Tool Wear Failure",4:"Random Failures"}
    st.write(status.get(predicted_class))
