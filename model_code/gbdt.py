from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

def model_gbdt(X_train,X_test,y_train,y_test):

    gbdt_model_base = GradientBoostingClassifier()

    gbdt_model_best = GradientBoostingClassifier(
        learning_rate=0.01,
        n_estimators=500,
        min_samples_split=90,
        min_samples_leaf=30,
        max_depth=8,
        # max_features=170,
        subsample=0.8
    )

    #调参后的模型
    gbdt_model = gbdt_model_best
    gbdt_model.fit(X_train,y_train)

    y_prob = gbdt_model.predict_proba(X_test)[:,1]

    return roc_auc_score(y_test,y_prob)

