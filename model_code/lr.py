from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def model_lr(X_train,X_test,y_train,y_test):

    lr_model_base = LogisticRegression()

    lr_model_best = LogisticRegression(
        C=0.1,
        penalty='l1',
        sovler='libnear'
    )

    #调参后的模型
    lr_model = lr_model_best
    lr_model.fit(X_train,y_train)
    y_prob = lr_model.predict_proba(X_test)[:,1]

    return roc_auc_score(y_test,y_prob)
