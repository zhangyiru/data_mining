from xgboost import XGBClassifier
import xgboost as xgb
import numpy as np

from sklearn.metrics import roc_auc_score

def model_lr(X_train,X_test,y_train,y_test):

    xgb_model_base = XGBClassifier()

    xgb_model_best = XGBClassifier(
        learning_rate = 0.1
    )

    #调参后的模型
    xgb_model = xgb_model_best
    xgb_model.fit(X_train,y_train)
    y_prob = xgb_model.predict_proba(X_test)[:,1]
    y_pred = np.where(y_prob>0.5,1,0)

    return roc_auc_score(y_test,y_pred)
