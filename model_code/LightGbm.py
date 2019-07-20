#!/usr/bin/env python
# -*- coding:utf-8 -*-
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

def model_lgb(X_train,X_test,y_train,y_test):
    #创建成lgb特征的数据集格式
    lgb_train = lgb.Dataset(X_train,y_train)
    lgb_eval = lgb.Dataset(X_test,y_test,reference = lgb_train)

    params = {
        'n_estimator':30,
        'task':'train',
        'boosting_type':'gbdt',
        'objective':'regression',
        'metric':{'l2','auc'},
        'num_leaves':31,
        'learning_rate':0.05,
        'feature_fraction':0.9,
        'bagging_fraction':0.8,
        'bagging_freq':5,
        'verbose':0
    }

    #训练
    lgb_model = lgb.train(
        params,
        lgb_train,
        num_boost_round=20,
        valid_sets=lgb_eval,
        early_stopping_rounds=5
    )

    #预测数据集
    y_pred = lgb_model.predict(
        X_test,
        num_iteration=lgb_model.best_iteration
    )

    return roc_auc_score(y_test,y_pred)