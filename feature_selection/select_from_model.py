#!/usr/bin/env python
# -*- coding:utf-8 -*-
from sklearn.feature_selection import SelectFromModel

"""基于模型的特征选择"""
def sfm(model,X_train,X_test):
    i=0
    select_model = SelectFromModel(model,prefit=True)
    X_train_new = select_model.transform(X_train)
    X_test_new = select_model.transform(X_test)

    if i==0:
        print("新训练特征维度：",X_train_new.shape)
    i+=1

    return X_train_new,X_test_new