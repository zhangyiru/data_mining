from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from feature_selection.select_from_model import sfm


def model_rf(X_train,X_test,y_train,y_test,feature_selection=False):

    rf_model_base = RandomForestClassifier()

    rf_model_best = RandomForestClassifier(
        oob_score=True,
        n_estimators=52,
        max_depth=13,
        min_samples_split=20,
        min_samples_leaf=12,
        # max_features=150,
        max_leaf_nodes=900
    )

    #调参后的模型
    rf_model = rf_model_best
    rf_model.fit(X_train,y_train)

    #基于模型的特性选择
    if feature_selection==True:
        X_train_new , X_test_new = sfm(rf_model,X_train,X_test)
        rf_model.fit(X_train_new,y_train)
        y_prob = rf_model.predict_proba(X_test_new)[:,1]

    else:
        y_prob = rf_model.predict_proba(X_test)[:,1]

    return roc_auc_score(y_test,y_prob)

