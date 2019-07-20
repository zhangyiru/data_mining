#!/usr/bin/env python
# -*- coding:utf-8 -*-
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

def model_svm(X_train,X_test,y_train,y_test):

    svm = SVC(kernel='linear',C=1)
    svm.fit(X_train,y_train)
    y_pred = svm.predict(X_test)

    return roc_auc_score(y_test,y_pred)