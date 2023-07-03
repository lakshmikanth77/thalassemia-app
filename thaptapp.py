import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def train_model(data_path):
    df = pd.read_csv("C:\\Users\\klkan\\Downloads\\thdata_final2.csv")
    X = df.drop('Group', axis=1)
    y = df['Group']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(n_estimators=100, min_samples_split=2, min_samples_leaf= 4, max_features= 'sqrt' , max_depth= 10, random_state=42)
    model.fit(X_train, y_train)
    
    return model

def run():
    st.title("Thalassemia Patient class Prediction")
    st.sidebar.header('Input patient biomarkers')

    def user_input_features():
        age = st.sidebar.slider('Age', 15, 55, 15)
        hb = st.sidebar.slider('Hb (g/dL)', 7.0, 18.0, 7.0)
        mch = st.sidebar.slider('MCH (pg)', 15.0, 34.0, 15.0)
        mchc = st.sidebar.slider('MCHC (g/dL)', 25.0, 37.0, 25.0)
        rdw = st.sidebar.slider('RDW (%)', 9.0, 27.0, 9.0)
        rbc_count = st.sidebar.slider('RBC count (million cells/mcL)', 3.0, 8.0, 3.0)
        data = {
            'Age': age,
            'Hb': hb,
            'MCH': mch,
            'MCHC': mchc,
            'RDW': rdw,
            'RBC count': rbc_count
        }
        return data
    
    data = user_input_features()
    data_df = pd.DataFrame(data, index=[0])
    
    st.subheader('Patient biomarkers')
    st.write(data_df)
    
    # Button to make prediction
    if st.button('Predict Group'):
        model = train_model("C:\\Users\\klkan\\Downloads\\thdata_final2.csv")
        prediction = model.predict(data_df)
        prediction_proba = model.predict_proba(data_df)
        
        st.subheader('Prediction')
        group_dict = {1: 'Normal or no Thalassemia present',
                      2: 'Alpha Thalassemia minor (milder form)',
                      3: 'Hbh disease (requires regular blood transfusion)',
                      4: 'Alpha Thalassemia major (Life threatening)'}
        predicted_group = group_dict[prediction[0]]
        confidence = prediction_proba[0][prediction[0]-1]
        
        color_dict = {1: 'green',
                      2: 'blue',
                      3: 'orange',
                      4: 'red'}
        color = color_dict[prediction[0]]
        
        st.markdown(f"<h2 style='text-align: center; color: white; background-color: {color};'>Predicted Group: {prediction[0]} - {predicted_group}</h2>", unsafe_allow_html=True)
        st.write(f'Confidence in prediction: {confidence*100:.2f}%')

run()
