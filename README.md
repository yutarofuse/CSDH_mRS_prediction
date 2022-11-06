# CSDH_mRS_prediction
Python scripts of the machine learning models' outcome prediction for the patients with CSDH.
The current work is in  "Development of machine learning models for predicting unfavorable functional outcomes in patients with chronic subdural hematomas."
'SelectPercentile.py': Comparison of the four machine learning models on the features selected by a filter method.
'SelectPercentile+SMOTE7':Comparison of the four machine learning models on the features selected by a filter method. In addition, an over-sampling technique is used.
'SelectPercentile+CNN':Comparison of the four machine learning models on the features selected by a filter method. In addition, an under-sampling technique is used.
'RFE7.py': Comparison of the four machine learning models on the features selected by a wrapper method.
'RFE+SMOTE7.py': Comparison of the four machine learning models on the features selected by a wrapper method.In addition, an over-sampling technique is used.
'RFE+CNN.py': Comparison of the four machine learning models on the features selected by a wrapper method.In addition, an under-sampling technique is used.
'Results.py': External validation of the best-performing machine learning models with the optimal hyperparameters.
