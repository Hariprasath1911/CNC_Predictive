import streamlit as st
import pandas as pd
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LeakyReLU

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
Tpe_map={'Low':1,'High':0,'Medium':2}
Type=Type_map.get(Type_select)
Air_Temperature = st.number_input('Enter Air Temperature', min_value=0)
Process_temperature = st.number_input('Enter Process temperature',min_value=0)
Rotational_speed_1= st.number_input('Enter Rotational speed',min_value=0)
Rotational_speed=1/(Rotational_speed_1**2)
Torque= st.number_input('Enter Torque',min_value=0)
Tool_wear=st.number_input('Enter Tool wear',min_value=0)

if st.button('Predict'):
    import numpy as np
    c1=np.array([[Type,Air_Temperature,Process_temperature,Rotational_speed,Torque,Tool_wear]])
    c2=scaler.transform(c1)
    predicted_output = model.predict(c2)
    predicted_class = np.argmax(predicted_output, axis=1)
    st.write("Job Status:", predicted_class)


