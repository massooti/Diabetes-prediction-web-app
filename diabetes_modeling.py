## This file is provided for easier readability of the code.
# There is no difference in functionality compared to the code in the Jupyter notebook.

from pycaret.utils import check_metric
from pycaret.regression import *
import pandas as pd
from pycaret.classification import *
import streamlit as st
from sklearn.model_selection import train_test_split
import xgboost as xgb
from pycaret.datasets import get_data

dataset = get_data('diabetes')
dataset.describe()
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
# best_model


rf = create_model('rf')
tuned_rf = tune_model(rf)
# finalize rf model
final_rf = finalize_model(tuned_rf)
predict_model(final_rf)
unseen_predictions = predict_model(final_rf, data=data_unseen)
unseen_predictions.head()


# check metric on unseen data
check_metric(unseen_predictions['Outcome'],
             unseen_predictions['prediction_label'], metric='Accuracy')

plot_model(tuned_rf, plot='confusion_matrix')
evaluate_model(tuned_rf)
save_model(final_rf, model_name='final_model_rf')


xgboost = create_model('xgboost')
tuned_xgboost = tune_model(xgboost)
plot_model(tuned_xgboost, plot='auc')
# finalize rf model
final_xg = finalize_model(tuned_xgboost)
# predict on new data
unseen_predictions_xg = predict_model(final_xg)
# unseen_predictions_xg.head()


# check metric on unseen data
check_metric(unseen_predictions_xg['Outcome'],
             unseen_predictions_xg['prediction_label'], metric='Accuracy')

plot_model(tuned_xgboost, plot='confusion_matrix')
evaluate_model(tuned_xgboost)

save_model(final_xg, model_name='final_model_xgb')
