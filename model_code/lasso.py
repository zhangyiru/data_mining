from sklearn.linear_model import Lasso
from sklearn.metrics import roc_auc_score

def model_lasso(X_train,X_test,y_train,y_test):

    lasso_model_base = Lasso()

    lasso_model_best = Lasso(
        alpha=0.001
    )

    #调参后的模型
    lasso_model = lasso_model_best
    lasso_model.fit(X_train,y_train)
    y_prob = lasso_model.predict(X_test)

    return roc_auc_score(y_test,y_prob)

