from pycaret.utils import check_metric
from pycaret.regression import *
import pandas as pd
from pycaret.classification import *
import streamlit as st
from sklearn.model_selection import train_test_split
from pycaret.datasets import get_data
from matplotlib import pyplot as plt

st.title("Impelementing RandomForest & XGBoost on Diabetes Dataset using Pycaret")
dataset = get_data('diabetes')


def rf_modeling():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    rf = create_model('rf')
    tuned_rf = tune_model(rf)
    plot_model(tuned_rf, plot='auc',
               display_format='streamlit', save=True)

    if st.checkbox("if the plot is not shown click: ", value=False):
        st.image("AUC.png")
    # finalize rf model
    final_rf = finalize_model(tuned_rf)
    predict_model(final_rf)
    unseen_predictions = predict_model(final_rf, data=data_unseen)
    unseen_predictions.head()
    chk = check_metric(unseen_predictions['Outcome'],
                       unseen_predictions['prediction_label'], metric='Accuracy')
    st.write(chk)
    rf_cnfmtrx = plot_model(tuned_rf, plot='confusion_matrix')
    st.pyplot(rf_cnfmtrx)
    evaluate_model(tuned_rf)
    # st.write(eve)


def xgb_modeling():
    xgboost = create_model('xgboost')
    tuned_xgboost = tune_model(xgboost)
    plot_model(tuned_xgboost, plot='auc')
    # finalize rf model
    final_xg = finalize_model(tuned_xgboost)
    # predict on new data
    unseen_predictions_xg = predict_model(final_xg, drift_report=True)

    # check metric on unseen data
    check_metric(unseen_predictions_xg['Outcome'],
                 unseen_predictions_xg['prediction_label'], metric='Accuracy')

    plot_model(tuned_xgboost, plot='confusion_matrix')
    evaluate_model(tuned_xgboost)


st.subheader('Machine learning model')
st.caption('برای دیدن نمای کلی دیتاست کلیک کنید')
if st.checkbox("show dataset overview: ", value=False):
    st.dataframe(dataset)


# sample 5% of data to be used as unseen data
data = dataset.sample(frac=0.95, random_state=786)
data_unseen = dataset.drop(data.index)
data.reset_index(inplace=True, drop=True)
data_unseen.reset_index(inplace=True, drop=True)

# print the revised shape
print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))
data.head()
# clf = setup(data, target='Outcome')
s = setup(data=data, target='Outcome', session_id=123)
best_model = compare_models(fold=5)
best_model
# print(best_model)

if st.checkbox("learn model Random Forest: ", value=False):
    rf_modeling()
elif st.checkbox("learn model XGBoost: ", value=False):
    xgb_modeling()
    # check metric on unseen data


# y = save_model(final_xg, 'Final XG Model 11Nov2020')


# # loading the saved model
# saved_final_rf = load_model('Final RF Model 11Nov2020')


# new_prediction = predict_model(saved_final_rf, data=data_unseen)
# new_prediction.head()


# o = check_metric(new_prediction['Outcome'],
#                  new_prediction['prediction_label'], metric='Accuracy')

# o
