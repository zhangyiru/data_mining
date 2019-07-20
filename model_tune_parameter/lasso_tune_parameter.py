#!/usr/bin/env python
# -*- coding:utf-8 -*-
from sklearn.linear_model import Lasso
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
import pandas as pd


#读取数据集
X_train = pd.read_csv("../tune_data/X_train.csv")
X_test = pd.read_csv("../tune_data/X_test.csv")
y_train = pd.read_csv("../tune_data/y_train.csv")
y_test = pd.read_csv("../tune_data/y_test.csv")

#转成array,再转成一维
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

lasso_model_base = Lasso()

lasso_model_best = Lasso(

)

tune_parameters = {
    'alpha':[0.001,0.01,0.1,1,10,100]
}
lasso = GridSearchCV(lasso_model_best,tune_parameters,scoring='roc_auc',cv=10)
lasso.fit(X_train,y_train)
print(lasso.best_params_)
print(lasso.best_score_)