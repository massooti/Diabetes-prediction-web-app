from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np


def predict_quality(model, df):
    predictions_data = predict_model(estimator=model, data=df)

    return predictions_data['prediction_label'][0]


def get_features_df():
    Pregnancies = st.sidebar.slider(label='Pregnancies', min_value=0,
                                    max_value=20,
                                    value=0,
                                    step=1)

    Glucose = st.sidebar.slider(label='Glucose', min_value=0.00,
                                max_value=199.00,
                                value=0.00,
                                step=0.01)

    BloodPressure = st.sidebar.slider(label='BloodPressure', min_value=0.00,
                                      max_value=125.00,
                                      value=0.00,
                                      step=0.01)

    SkinThickness = st.sidebar.slider(label='SkinThickness', min_value=0.0,
                                      max_value=100.0,
                                      value=8.0,
                                      step=0.1)

    Insulin = st.sidebar.slider(label='Insulin', min_value=0.000,
                                max_value=846.000,
                                value=0.500,
                                step=0.001)

    BMI = st.sidebar.slider(label='BMI', min_value=0.000,
                            max_value=72.00,
                            value=36.00,
                            step=1.00)

    DiabetesPedigreeFunction = st.sidebar.slider(label='DiabetesPedigreeFunction', min_value=0.078,
                                                 max_value=2.42,
                                                 value=0.001,
                                                 step=0.001)

    Age = st.sidebar.slider(label='Age', min_value=20,
                            max_value=81,
                            value=20,
                            step=1)

    features = {'Pregnancies': Pregnancies, 'Glucose': Glucose,
                'BloodPressure': BloodPressure, 'SkinThickness': SkinThickness,
                'Insulin': Insulin, 'BMI': BMI,
                'DiabetesPedigreeFunction': DiabetesPedigreeFunction, 'Age': Age,
                }
    return features, Age


def features_info() -> pd.DataFrame:
    infos = ['Pregnancies',
             'Glucose',
             'BloodPressure',
             'SkinThickness',
             'Insulin',
             'BMI',
             'DiabetesPedigreeFunction',
             'Age',]

    descri = ["Number of times pregnant",
              "Plasma glucose concentration a 2 hours in an oral glucose tolerance test",
              "Diastolic blood pressure (mm Hg)",
              "Triceps skin fold thickness (mm)",
              "2-Hour serum insulin (mu U/ml)",
              "Body mass index (weight in kg/(height in m)^2)",
              "Diabetes pedigree function",
              "Age (years)",]

    return pd.DataFrame({"name": infos, "desc": descri})


rf_model = load_model('final_model_rf')
xg_model = load_model('final_model_xgb')

st.title('Diabete predicting Web App')

st.caption(
    "This is a web app to predict diabetes based on several features that you can see in the _italics_: red[sidebar].Please adjust the value of each feature. After that, click on the Predict button at the bottom to see the prediction of the classifier.")

tab1, tab2, tab3, tab4 = st.tabs(["Random Forest", "XGBoost", "dataset overview", "author"])
features, sidebars = get_features_df()
features_df = pd.DataFrame([features])

with tab1:
    st.write(
        'In this tab you can predict patient situation based on Random Forest model')
    st.table(features_df)
    if st.button('Predict by RandomForest'):
        prediction = predict_quality(rf_model, features_df)
        st.title(
            'Based on feature values, your patient has diabets probability:' + str(prediction))

with tab2:
    st.write(
        'In this tab you can predict patient situation based on XGBOOST model')
    st.table(features_df)
    if st.button('Predict by XGBoost'):
        prediction = predict_quality(xg_model, features_df)
        st.title(
            'Based on feature values, your patient has diabets probability:' + str(prediction))

with tab3:
    dataset = pd.read_csv('diabetes.csv')
    st.caption("features explanation")
    st.table(features_info())
    st.caption("dataset overview")
    st.dataframe(dataset.head())


with tab4:
    st.title("Author: Masoud Zaeem")
    st.title("Email: massoudzaeem@gmail.com")