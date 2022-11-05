#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from numpy import sqrt
from numpy import argmax
import sklearn
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, ParameterGrid
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.compose import ColumnTransformer
from sklearn import set_config
from sklearn_pandas import DataFrameMapper
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import CondensedNearestNeighbour 

df = pd.read_csv('CSDH_internal.csv')
X = df.drop("RESULTS2",axis=1)
y = df["RESULTS2"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0, stratify = y)
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(X_train)
scaler.transform(X_train)
X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns)


# In[2]:


param_grid = {'max_iter' : [5000],  'class_weight' : ['balanced', '{{1:1}}']}
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring = 'roc_auc')
rfecv = RFECV(estimator=RandomForestClassifier(class_weight='balanced'), n_jobs=-1, scoring="accuracy", cv=5)
rfecv.fit(X_train, y_train)

X_train1 = rfecv.transform(X_train)
X_test1 = rfecv.transform(X_test)
cnn = CondensedNearestNeighbour(random_state=0) 
x_resampled, y_resampled = cnn.fit_sample(X_train1, y_train)
grid_search.fit(x_resampled, y_resampled)
Y_score = grid_search.predict_proba(X_test1)[:,1]
fpr, tpr, thresholds = roc_curve(y_true=y_test,y_score=Y_score)

J = tpr - fpr
ix = argmax(J)
best_thresh = thresholds[ix]
print('Best Threshold=%f, sensitivity = %.3f, specificity = %.3f, J=%.3f' % (best_thresh, tpr[ix], 1-fpr[ix], J[ix]))
y_prob_pred = (grid_search.predict_proba(X_test1)[:,1]>= best_thresh).astype(bool)

def specificity_score(y_test, y_prob_pred):
    tn, fp, fn, tp = confusion_matrix(y_test, y_prob_pred).flatten()
    return tn / (tn + fp)

f = open('percentile LogisticRegression.txt', 'a')
f.write('percentile LogisticRegression: {}'.format(rfecv.n_features_))
f.write("\n")
f = open('accuracy LogisticRegression.txt', 'a')
f.write(format(accuracy_score(y_true = y_test , y_pred = y_prob_pred)))
f.write("\n")
f = open('AUC LogisticRegression.txt', 'a')
f.write(format(roc_auc_score(y_true=y_test,y_score=Y_score)))
f.write("\n")
f.close()
f = open('AUC LogisticRegression best parameters.txt', 'a')
f.write(format(grid_search.best_params_))
f.write("\n")
f.close()
f = open('f1 score LogisticRegression.txt', 'a')
f.write(format(f1_score(y_true = y_test , y_pred = y_prob_pred)))
f.write("\n")
f = open('sensitivity LogisticRegression.txt', 'a')
f.write(format(recall_score(y_true = y_test , y_pred = y_prob_pred)))
f.write("\n")
f = open('specificity LogisticRegression.txt', 'a')
f.write(format(specificity_score(y_test , y_prob_pred)))
f.write("\n")
f = open('PPV LogisticRegression.txt', 'a')
f.write(format(precision_score(y_true = y_test , y_pred = y_prob_pred)))
f.write("\n")
f.close()
#ROC曲線を描き、AUCを算出
plt.plot(fpr,tpr,label='roc curve logistic regression(AUC= %0.3f)' % auc(fpr,tpr))
plt.plot([0,0,1], [0,1,1], linestyle='--', label='ideal line')
plt.plot([0, 1], [0, 1], linestyle='--', label='random prediction')
plt.legend()
plt.xlabel('false positive rate(FPR)')
plt.ylabel('true positive rate(TPR)')
plt.savefig("RFECNNROCLR.tif", format = "tiff", dpi = 300, bbox_inches = 'tight')
plt.show()
print('confusion matrix = \n', confusion_matrix(y_true = y_test, y_pred = y_prob_pred))


# In[3]:


param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],  'gamma' : [0.001, 0.01, 0.1, 1, 10, 100], 'probability':[True]}
grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring= make_scorer(roc_auc_score, greater_is_better=True, needs_proba=True))

rfecv = RFECV(estimator=RandomForestClassifier(class_weight='balanced'), n_jobs=-1, scoring="accuracy", cv=5)
rfecv.fit(X_train, y_train)
X_train1 = rfecv.transform(X_train)
X_test1 = rfecv.transform(X_test)
cnn = CondensedNearestNeighbour(random_state=0) 
x_resampled, y_resampled = cnn.fit_sample(X_train1, y_train)
grid_search.fit(x_resampled, y_resampled)
Y_score = grid_search.predict_proba(X_test1)[:,1]
fpr1, tpr1, thresholds = roc_curve(y_true=y_test,y_score=Y_score)

J = tpr1 - fpr1
ix = argmax(J)
best_thresh = thresholds[ix]
print('Best Threshold=%f, sensitivity = %.3f, specificity = %.3f, J=%.3f' % (best_thresh, tpr1[ix], 1-fpr1[ix], J[ix]))
y_prob_pred = (grid_search.predict_proba(X_test1)[:,1]>= best_thresh).astype(bool)

def specificity_score(y_test, y_prob_pred):
    tn, fp, fn, tp = confusion_matrix(y_test, y_prob_pred).flatten()
    return tn / (tn + fp)

f = open('percentile SVM.txt', 'a')
f.write('percentile SVM: {}'.format(rfecv.n_features_))
f.write("\n")
f = open('accuracy SVM.txt', 'a')
f.write(format(accuracy_score(y_true = y_test , y_pred = y_prob_pred)))
f.write("\n")
f = open('AUC SVM.txt', 'a')
f.write(format(roc_auc_score(y_true=y_test,y_score=Y_score)))
f.write("\n")
f.close()
f = open('AUC SVM best parameters.txt', 'a')
f.write(format(grid_search.best_params_))
f.write("\n")
f.close()
f = open('f1 score SVM.txt', 'a')
f.write(format(f1_score(y_true = y_test , y_pred = y_prob_pred)))
f.write("\n")
f = open('sensitivity SVM.txt', 'a')
f.write(format(recall_score(y_true = y_test , y_pred = y_prob_pred)))
f.write("\n")
f = open('specificity SVM.txt', 'a')
f.write(format(specificity_score(y_test , y_prob_pred)))
f.write("\n")
f = open('PPV SVM.txt', 'a')
f.write(format(precision_score(y_true = y_test , y_pred = y_prob_pred)))
f.write("\n")
f.close()
#ROC曲線を描き、AUCを算出
plt.plot(fpr1,tpr1,label='roc curve SVM(AUC= %0.3f)' % auc(fpr1,tpr1))
plt.plot([0,0,1], [0,1,1], linestyle='--', label='ideal line')
plt.plot([0, 1], [0, 1], linestyle='--', label='random prediction')
plt.legend()
plt.xlabel('false positive rate(FPR)')
plt.ylabel('true positive rate(TPR)')
plt.savefig("RFECNNROCSVM.tif", format = "tiff", dpi = 300, bbox_inches = 'tight')
plt.show()
print('confusion matrix = \n', confusion_matrix(y_true = y_test, y_pred = y_prob_pred))


# In[4]:


param_grid = {'criterion'   : ['gini'],'n_estimators': [1000],'max_features': ['sqrt'], 'min_samples_leaf': [2, 5, 10], 'min_samples_split': [2, 5], 'max_depth': [7, 63, 200], 'class_weight': ['balanced', None]}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring = 'roc_auc')

rfecv = RFECV(estimator=RandomForestClassifier(class_weight='balanced'), n_jobs=-1, scoring="accuracy", cv=5)
rfecv.fit(X_train, y_train)
X_train1 = rfecv.transform(X_train)
X_test1 = rfecv.transform(X_test)
cnn = CondensedNearestNeighbour(random_state=0) 
x_resampled, y_resampled = cnn.fit_sample(X_train1, y_train)
grid_search.fit(x_resampled, y_resampled)
Y_score = grid_search.predict_proba(X_test1)[:,1]
fpr2, tpr2, thresholds = roc_curve(y_true=y_test,y_score=Y_score)
# get the best threshold
J = tpr2 - fpr2
ix = argmax(J)
best_thresh = thresholds[ix]
print('Best Threshold=%f, sensitivity = %.3f, specificity = %.3f, J=%.3f' % (best_thresh, tpr2[ix], 1-fpr2[ix], J[ix]))
y_prob_pred = (grid_search.predict_proba(X_test1)[:,1]>= best_thresh).astype(bool)

def specificity_score(y_test, y_prob_pred):
    tn, fp, fn, tp = confusion_matrix(y_test, y_prob_pred).flatten()
    return tn / (tn + fp)

f = open('percentile RF.txt', 'a')
f.write('percentile RF: {}'.format(rfecv.n_features_))
f.write("\n")
f = open('accuracy RF.txt', 'a')
f.write(format(accuracy_score(y_true = y_test , y_pred = y_prob_pred)))
f.write("\n")
f = open('AUC RF.txt', 'a')
f.write(format(roc_auc_score(y_true=y_test,y_score=Y_score)))
f.write("\n")
f.close()
f = open('AUC RF best parameters.txt', 'a')
f.write(format(grid_search.best_params_))
f.write("\n")
f.close()
f = open('f1 score RF.txt', 'a')
f.write(format(f1_score(y_true = y_test , y_pred = y_prob_pred)))
f.write("\n")
f = open('sensitivity RF.txt', 'a')
f.write(format(recall_score(y_true = y_test , y_pred = y_prob_pred)))
f.write("\n")
f = open('specificity RF.txt', 'a')
f.write(format(specificity_score(y_test , y_prob_pred)))
f.write("\n")
f = open('PPV RF.txt', 'a')
f.write(format(precision_score(y_true = y_test , y_pred = y_prob_pred)))
f.write("\n")
f.close()
#ROC曲線を描き、AUCを算出
plt.plot(fpr2,tpr2,label='roc curve RF(AUC= %0.3f)' % auc(fpr2,tpr2))
plt.plot([0,0,1], [0,1,1], linestyle='--', label='ideal line')
plt.plot([0, 1], [0, 1], linestyle='--', label='random prediction')
plt.legend()
plt.xlabel('false positive rate(FPR)')
plt.ylabel('true positive rate(TPR)')
plt.savefig("RFECNNROCRF.tif", format = "tiff", dpi = 300, bbox_inches = 'tight')
plt.show()
print('confusion matrix = \n', confusion_matrix(y_true = y_test, y_pred = y_prob_pred))
plt.clf()
feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances
X_train1 = pd.DataFrame(X_train1)
fi = feature_importances  
fi_df = pd.DataFrame({'feature': list(X_train1.columns), 'feature importance': fi[:]}).sort_values('feature importance', ascending = False)
fi_df
sns.barplot(fi_df['feature importance'],fi_df['feature'])
plt.savefig("RFECNNfeatureimportanceRF.tif", format= "tiff", dpi = 300, bbox_inches = 'tight')
plt.clf()


# In[5]:


import lightgbm as lgb
param_grid = {'num_leaves': [7, 15, 31], 'learning_rate': [0.1, 0.01, 0.005], 'feature_fraction': [0.5, 0.8],'bagging_fraction': [0.8], 'bagging_freq': [1, 3]}

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0, stratify = y)

scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(X_train)
scaler.transform(X_train)
X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns)
lgb_estimator = lgb.LGBMClassifier(boosting_type='gbdt', objective='binary',  verbosity = -1, metric='auc', random_state = 0)
grid_search = GridSearchCV(lgb_estimator, param_grid, cv=5, scoring = 'roc_auc')
rfecv = RFECV(estimator=RandomForestClassifier(class_weight='balanced'), n_jobs=-1, scoring="accuracy", cv=5)
rfecv.fit(X_train, y_train)
X_train1 = rfecv.transform(X_train)
X_test1 = rfecv.transform(X_test)
cnn = CondensedNearestNeighbour(random_state=0) 
x_resampled, y_resampled = cnn.fit_sample(X_train1, y_train)
grid_search.fit(x_resampled, y_resampled)
Y_pred = grid_search.predict(X_test1)
Y_score = grid_search.predict_proba(X_test1)[:,1]
fpr4, tpr4, thresholds = roc_curve(y_true=y_test,y_score=Y_score)
# get the best threshold
J = tpr4 - fpr4
ix = argmax(J)
best_thresh = thresholds[ix]
print('Best Threshold=%f, sensitivity = %.3f, specificity = %.3f, J=%.3f' % (best_thresh, tpr4[ix], 1-fpr4[ix], J[ix]))
y_prob_pred = (grid_search.predict_proba(X_test1)[:,1]>= best_thresh).astype(bool)

def specificity_score(y_test, y_prob_pred):
    tn, fp, fn, tp = confusion_matrix(y_test, y_prob_pred).flatten()
    return tn / (tn + fp)

f = open('percentile LGBM.txt', 'a')
f.write('percentile LGBM: {}'.format(rfecv.n_features_))
f.write("\n")
f = open('accuracy LGBM.txt', 'a')
f.write(format(accuracy_score(y_true = y_test , y_pred = y_prob_pred)))
f.write("\n")
f = open('AUC LGBM.txt', 'a')
f.write(format(roc_auc_score(y_true=y_test,y_score=Y_score)))
f.write("\n")
f.close()
f = open('AUC LGBM best parameters.txt', 'a')
f.write(format(grid_search.best_params_))
f.write("\n")
f.close()
f = open('f1 score LGBM.txt', 'a')
f.write(format(f1_score(y_true = y_test , y_pred = y_prob_pred)))
f.write("\n")
f = open('sensitivity LGBM.txt', 'a')
f.write(format(recall_score(y_true = y_test , y_pred = y_prob_pred)))
f.write("\n")
f = open('specificity LGBM.txt', 'a')
f.write(format(specificity_score(y_test , y_prob_pred)))
f.write("\n")
f = open('PPV LGBM.txt', 'a')
f.write(format(precision_score(y_true = y_test , y_pred = y_prob_pred)))
f.write("\n")
f.close()
#ROC曲線を描き、AUCを算出
plt.plot(fpr4,tpr4,label='roc curve LGBM(AUC= %0.3f)' % auc(fpr4,tpr4))
plt.plot([0,0,1], [0,1,1], linestyle='--', label='ideal line')
plt.plot([0, 1], [0, 1], linestyle='--', label='random prediction')
plt.legend()
plt.xlabel('false positive rate(FPR)')
plt.ylabel('true positive rate(TPR)')
plt.savefig("RFECNNROCLGBM.tif", format = "tiff", dpi = 300, bbox_inches = 'tight')
plt.show()
plt.clf()
print('confusion matrix = \n', confusion_matrix(y_true = y_test, y_pred = y_prob_pred))

feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances
X_train1 = pd.DataFrame(X_train1)
fi = feature_importances  
fi_df = pd.DataFrame({'feature': list(X_train1.columns), 'feature importance': fi[:]}).sort_values('feature importance', ascending = False)
fi_df
sns.barplot(fi_df['feature importance'],fi_df['feature'])
plt.savefig("RFECNNfeatureimportanceLGBM.tif", format= "tiff", dpi = 300, bbox_inches = 'tight')
plt.clf()


# In[ ]:




