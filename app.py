import streamlit as st
import joblib 
import numpy as np

st.write("""
# GRE addmission prediction app
This app predict **GRE addmission chance**
""")



st.sidebar.header('Specify the input parameters')

def user_input_features():
    gre=st.sidebar.slider("enter the gre score in range 0 to 1" , min_value=0.0,max_value=1.0,value=0.5)
    tofel=st.sidebar.slider("enter the tofel score in range 0 to 1", min_value=0.0,max_value=1.0,value=0.5)
    University_R=st.sidebar.slider("enter the university rating in range 0 to 1", min_value=0.0, max_value=1.0,value=0.5)
    SOP=st.sidebar.slider(
        "enter the SOP in range 0 to 1", min_value=0.0, max_value=1.0,value=0.5)
    CGPA=st.sidebar.slider(
        "enter the CGPA in range 0 to 1", min_value=0.0, max_value=1.0,value=0.5)
    
    feature =np.array([[gre,tofel,University_R,SOP,CGPA]])
    return feature

feature=user_input_features()

st.header("specified input parameters")
st.write(feature)
st.write('----')
predictor = joblib.load('regressor.joblib')
prediction=predictor.predict(feature)
st.header('chance of addmission is ')
st.write(prediction)
