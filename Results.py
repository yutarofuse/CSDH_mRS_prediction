#!/usr/bin/env python
# coding: utf-8

# In[25]:


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

df2 = pd.read_csv('CSDH_external.csv')

X2 = df2.drop("RESULTS2",axis=1)
y2 = df2["RESULTS2"]


# In[26]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0, stratify = y)

scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(X_train)
scaler.transform(X_train)
X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns)
X_test2 = pd.DataFrame(scaler.transform(X2), columns=X_train.columns)


# In[27]:


select = SelectPercentile(percentile = 20)
select.fit(X_train, y_train)
select.transform(X_train)
X_train1 = X_train.loc[:, select.get_support()]
X_test1 = X_test.loc[:, select.get_support()]
X_test2_1 = X_test2.loc[:, select.get_support()]

param_grid = {'max_iter' : [5000],  'class_weight' : [{1:1}]}
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring = 'roc_auc')
grid_search.fit(X_train1, y_train)
Y_score = grid_search.predict_proba(X_test1)[:,1]
Y_score2 = grid_search.predict_proba(X_test2_1)[:,1]
fpr1, tpr1, thresholds = roc_curve(y_true=y_test,y_score=Y_score)
fpr2, tpr2, thresholds2 = roc_curve(y_true=y2,y_score=Y_score2)

# get the best threshold
J = tpr1 - fpr1
ix = argmax(J)
best_thresh = thresholds[ix]
print('Best Threshold=%f, sensitivity = %.3f, specificity = %.3f, J=%.3f' % (best_thresh, tpr1[ix], 1-fpr1[ix], J[ix]))
y_prob_pred = (grid_search.predict_proba(X_test1)[:,1]>= best_thresh).astype(bool)
print(classification_report(y_test, y_prob_pred, target_names=['no DHN', 'DHN'], digits = 3))
print('confusion matrix = \n', confusion_matrix(y_true = y_test, y_pred = y_prob_pred))
J = tpr2 - fpr2
ix = argmax(J)
best_thresh = thresholds2[ix]
print('Best Threshold=%f, sensitivity = %.3f, specificity = %.3f, J=%.3f' % (best_thresh, tpr2[ix], 1-fpr2[ix], J[ix]))
y_prob_pred2 = (grid_search.predict_proba(X_test2_1)[:,1]>= best_thresh).astype(bool)
print(classification_report(y2, y_prob_pred2, target_names=['no DHN', 'DHN'], digits = 3))
print('confusion matrix = \n', confusion_matrix(y_true =  y2 , y_pred = y_prob_pred2))

#ROC曲線を描き、AUCを算出
plt.plot(fpr1,tpr1,label='roc curve logistic regression(AUC= %0.3f)' % auc(fpr1,tpr1))
plt.plot(fpr2,tpr2,label='roc curve logistic regression ex(AUC= %0.3f)' % auc(fpr2,tpr2))
plt.plot([0,0,1], [0,1,1], linestyle='--', label='ideal line')
plt.plot([0, 1], [0, 1], linestyle='--', label='random prediction')
plt.legend()
plt.xlabel('false positive rate(FPR)')
plt.ylabel('true positive rate(TPR)')
plt.show()


# In[28]:


coef = grid_search.best_estimator_.coef_
df_co = pd.DataFrame(coef.reshape((11, -1)), X_train1.columns, columns={"coefficient"})
interce = pd.DataFrame([grid_search.best_estimator_.intercept_], index=["constant"], columns=["coefficient"])
df_coef = pd.concat([df_co, interce])
df_coef.to_csv("LR_coef.csv")
#df_coef.to_csv("LR_coef.csv")


# In[29]:


df = pd.read_csv('LR_coef3.csv')
df
df = df.rename(columns = {'Features': 'feature'})
#df = df.sort_values(['Standardized Beta Coefficient'], ascending=False).reset_index(drop=True)


# In[30]:


sns.barplot(df['Standardized Beta Coefficient'],df['feature'], order=df.sort_values(by=['Standardized Beta Coefficient'], ascending=False).set_index('feature').index,orient = "h")
plt.savefig("Coefficient.tif", format= "tiff", dpi = 1200, bbox_inches = 'tight')


# In[31]:


param_grid = {'C': [0.001],  'gamma' : [0.001], 'probability':[True]}
grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring= make_scorer(roc_auc_score, greater_is_better=True, needs_proba=True))

select = SelectPercentile(percentile = 10)
select.fit(X_train, y_train)
X_train1 = select.transform(X_train)
X_test1 = select.transform(X_test)
X_train1 = X_train.loc[:, select.get_support()]
X_test1 = X_test.loc[:, select.get_support()]
X_test2_1 = X_test2.loc[:, select.get_support()]

#sm = SMOTE()
#x_resampled, y_resampled = sm.fit_sample(X_train1, y_train)
#grid_search.fit(x_resampled, y_resampled)
grid_search.fit(X_train1, y_train)
Y_score = grid_search.predict_proba(X_test1)[:,1]
Y_score2 = grid_search.predict_proba(X_test2_1)[:,1]
fpr3, tpr3, thresholds = roc_curve(y_true=y_test,y_score=Y_score)
fpr4, tpr4, thresholds2 = roc_curve(y_true=y2,y_score=Y_score2)

# get the best threshold
J = tpr3 - fpr3
ix = argmax(J)
best_thresh = thresholds[ix]
print('Best Threshold=%f, sensitivity = %.3f, specificity = %.3f, J=%.3f' % (best_thresh, tpr3[ix], 1-fpr3[ix], J[ix]))
y_prob_pred = (grid_search.predict_proba(X_test1)[:,1]>= best_thresh).astype(bool)
print(classification_report(y_test, y_prob_pred, target_names=['normal', 'hypo'], digits = 3))
print('confusion matrix = \n', confusion_matrix(y_true = y_test, y_pred = y_prob_pred))
# get the best threshold
J = tpr4 - fpr4
ix = argmax(J)
best_thresh = thresholds2[ix]
print('Best Threshold=%f, sensitivity = %.3f, specificity = %.3f, J=%.3f' % (best_thresh, tpr4[ix], 1-fpr4[ix], J[ix]))

y_prob_pred2 = (grid_search.predict_proba(X_test2_1)[:,1]>= best_thresh).astype(bool)
print(classification_report(y2, y_prob_pred2, target_names=['normal', 'hypo'], digits = 3))
print('confusion matrix = \n', confusion_matrix(y_true =  y2 , y_pred = y_prob_pred2))

#ROC曲線を描き、AUCを算出
plt.plot(fpr3,tpr3,label='roc curve SVC(AUC= %0.3f)' % auc(fpr3,tpr3))
plt.plot(fpr4,tpr4,label='roc curve SVC ex(AUC= %0.3f)' % auc(fpr4,tpr4))
plt.plot([0,0,1], [0,1,1], linestyle='--', label='ideal line')
plt.plot([0, 1], [0, 1], linestyle='--', label='random prediction')
plt.legend()
plt.xlabel('false positive rate(FPR)')
plt.ylabel('true positive rate(TPR)')
plt.show()


# In[58]:


param_grid = {'criterion'   : ['gini'],'n_estimators': [1000],'max_features': ['sqrt'], 'min_samples_leaf': [2], 'min_samples_split': [5], 'max_depth': [200], 'class_weight': [None]}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring = 'roc_auc')

select = SelectPercentile(percentile = 15)
select.fit(X_train, y_train)
select.transform(X_train)
X_train1 = X_train.loc[:, select.get_support()]
X_test1 = X_test.loc[:, select.get_support()]
X_test2_1 = X_test2.loc[:, select.get_support()]

sm = SMOTE(random_state = 0)
x_resampled, y_resampled = sm.fit_sample(X_train1, y_train)
grid_search.fit(x_resampled, y_resampled)
#grid_search.fit(X_train1, y_train)
Y_score = grid_search.predict_proba(X_test1)[:,1]
Y_score2 = grid_search.predict_proba(X_test2_1)[:,1]
fpr5, tpr5, thresholds = roc_curve(y_true=y_test,y_score=Y_score)
fpr6, tpr6, thresholds2 = roc_curve(y_true=y2,y_score=Y_score2)

# get the best threshold
J = tpr5 - fpr5
ix = argmax(J)
best_thresh = thresholds[ix]
print('Best Threshold=%f, sensitivity = %.3f, specificity = %.3f, J=%.3f' % (best_thresh, tpr5[ix], 1-fpr5[ix], J[ix]))
y_prob_pred = (grid_search.predict_proba(X_test1)[:,1]>= best_thresh).astype(bool)
print(classification_report(y_test, y_prob_pred, target_names=['normal', 'hypo'], digits = 3))
print('confusion matrix = \n', confusion_matrix(y_true = y_test, y_pred = y_prob_pred))
# get the best threshold
J = tpr6 - fpr6
ix = argmax(J)
best_thresh = thresholds2[ix]
print('Best Threshold=%f, sensitivity = %.3f, specificity = %.3f, J=%.3f' % (best_thresh, tpr6[ix], 1-fpr6[ix], J[ix]))
y_prob_pred2 = (grid_search.predict_proba(X_test2_1)[:,1]>= best_thresh).astype(bool)
print(classification_report(y2, y_prob_pred2, target_names=['normal', 'hypo'], digits = 3))
print('confusion matrix = \n', confusion_matrix(y_true =  y2 , y_pred = y_prob_pred2))

#ROC曲線を描き、AUCを算出
plt.plot(fpr5,tpr5,label='roc curve RF(AUC= %0.3f)' % auc(fpr5,tpr5))
plt.plot(fpr6,tpr6,label='roc curve RF ex(AUC= %0.3f)' % auc(fpr6,tpr6))
plt.plot([0,0,1], [0,1,1], linestyle='--', label='ideal line')
plt.plot([0, 1], [0, 1], linestyle='--', label='random prediction')
plt.legend()
plt.xlabel('false positive rate(FPR)')
plt.ylabel('true positive rate(TPR)')
plt.show()


# In[59]:


feature_importances = grid_search.best_estimator_.feature_importances_

feature_importances

X_train1 = pd.DataFrame(X_train1)
fi = feature_importances  
fi_df = pd.DataFrame({'feature': list(X_train1.columns), 'feature importance': fi[:]}).sort_values('feature importance', ascending = False)
fi_df
    
sns.barplot(fi_df['feature importance'],fi_df['feature'] ,orient = "h", color = 'gray')
plt.savefig("featureimportanceRF.tif", format= "tiff", dpi = 1200, bbox_inches = 'tight')


    
fi_df.to_csv("RF_fi.csv")


# In[60]:


import lightgbm as lgb

param_grid = {'bagging_fraction': [0.8], 'bagging_freq': [1], 'feature_fraction': [0.8], 'learning_rate': [0.01], 'num_leaves': [7]}
lgb_estimator = lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', verbosity = -1, metric='auc', random_state = 0)
grid_search = GridSearchCV(lgb_estimator, param_grid, cv=5, scoring = 'roc_auc')

select = SelectPercentile(percentile = 20)
select.fit(X_train, y_train)
select.transform(X_train)
X_train1 = X_train.loc[:, select.get_support()]
X_test1 = X_test.loc[:, select.get_support()]
X_test2_1 = X_test2.loc[:, select.get_support()]
cnn = CondensedNearestNeighbour(random_state=0) 
x_resampled, y_resampled = cnn.fit_sample(X_train1, y_train)
grid_search.fit(x_resampled, y_resampled)
Y_score = grid_search.predict_proba(X_test1)[:,1]
Y_score2 = grid_search.predict_proba(X_test2_1)[:,1]
fpr7, tpr7, thresholds = roc_curve(y_true=y_test,y_score=Y_score)
fpr8, tpr8, thresholds2 = roc_curve(y_true=y2,y_score=Y_score2)

# get the best threshold
J = tpr7 - fpr7
ix = argmax(J)
best_thresh = thresholds[ix]
print('Best Threshold=%f, sensitivity = %.3f, specificity = %.3f, J=%.3f' % (best_thresh, tpr7[ix], 1-fpr7[ix], J[ix]))
y_prob_pred = (grid_search.predict_proba(X_test1)[:,1]>= best_thresh).astype(bool)
print(classification_report(y_test, y_prob_pred, target_names=['normal', 'hypo'], digits = 3))
print('confusion matrix = \n', confusion_matrix(y_true = y_test, y_pred = y_prob_pred))
# get the best threshold
J = tpr8 - fpr8
ix = argmax(J)
best_thresh = thresholds2[ix]
print('Best Threshold=%f, sensitivity = %.3f, specificity = %.3f, J=%.3f' % (best_thresh, tpr8[ix], 1-fpr8[ix], J[ix]))
y_prob_pred2 = (grid_search.predict_proba(X_test2_1)[:,1]>= best_thresh).astype(bool)
print(classification_report(y2, y_prob_pred2, target_names=['normal', 'hypo'], digits = 3))
print('confusion matrix = \n', confusion_matrix(y_true =  y2 , y_pred = y_prob_pred2))

#ROC曲線を描き、AUCを算出
plt.plot(fpr7,tpr7,label='roc curve LGBM(AUC= %0.3f)' % auc(fpr7,tpr7))
plt.plot(fpr8,tpr8,label='roc curve LGBM ex(AUC= %0.3f)' % auc(fpr8,tpr8))
plt.plot([0,0,1], [0,1,1], linestyle='--', label='ideal line')
plt.plot([0, 1], [0, 1], linestyle='--', label='random prediction')
plt.legend()
plt.xlabel('false positive rate(FPR)')
plt.ylabel('true positive rate(TPR)')
plt.show()


# In[61]:


feature_importances2 = grid_search.best_estimator_.feature_importances_

feature_importances2

X_train1 = pd.DataFrame(X_train1)
fi2 = feature_importances2  
fi_df2 = pd.DataFrame({'feature': list(X_train1.columns), 'feature importance': fi2[:]}).sort_values('feature importance', ascending = False)
fi_df2

sns.barplot(fi_df2['feature importance'],fi_df2['feature'],orient = "h", color = 'gray' )
plt.savefig("featureimportanceLGBM.tif", format= "tiff", dpi = 1200, bbox_inches = 'tight')

fi_df2.to_csv("LGBM_fi.csv")


# In[62]:


#ROC曲線を描き、AUCを算出
plt.plot(fpr2,tpr2,label='Logistic regression (AUC= %0.3f)' % auc(fpr2,tpr2))
plt.plot(fpr4,tpr4,label='SVM (AUC= %0.3f)' % auc(fpr4,tpr4))
plt.plot(fpr6,tpr6,label='Random forest (AUC= %0.3f)' % auc(fpr6,tpr6))
plt.plot(fpr8,tpr8,label='Light GBM (AUC= %0.3f)' % auc(fpr8,tpr8))
plt.plot([0,0,1], [0,1,1], linestyle='--', color = 'gray')
plt.plot([0, 1], [0, 1], linestyle='--', color = 'gray')
plt.legend()
plt.xlabel('false positive rate (FPR)')
plt.ylabel('true positive rate (TPR)')
plt.savefig("ROC_ex.tif", format= "tiff", dpi = 1200, bbox_inches = 'tight')
plt.show()


# In[63]:


#ROC曲線を描き、AUCを算出
plt.plot(fpr1,tpr1,label='Logistic regression (AUC= %0.3f)' % auc(fpr1, tpr1))
plt.plot(fpr3,tpr3,label='SVM (AUC= %0.3f)' % auc(fpr3,tpr3))
plt.plot(fpr5,tpr5,label='Random forest (AUC= %0.3f)' % auc(fpr5,tpr5))
plt.plot(fpr7,tpr7,label='Light GBM (AUC= %0.3f)' % auc(fpr7,tpr7))
plt.plot([0,0,1], [0,1,1], linestyle='--', color = 'gray')
plt.plot([0, 1], [0, 1], linestyle='--', color = 'gray')
plt.legend()
plt.xlabel('false positive rate (FPR)')
plt.ylabel('true positive rate (TPR)')
plt.savefig("ROC.tif", format= "tiff", dpi = 1200, bbox_inches = 'tight')
plt.show()


# In[64]:


sns.barplot(df['Standardized Beta Coefficient'],df['feature'], order=df.sort_values(by=['Standardized Beta Coefficient'], ascending=False).set_index('feature').index, color = 'gray', orient = "h")
plt.savefig("Coefficient2.tif", format= "tiff", dpi = 1200, bbox_inches = 'tight')


# In[ ]:





# In[ ]:




