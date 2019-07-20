from sklearn.linear_model import LogisticRegression
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

lr_model_base = LogisticRegression()

lr_model_best = LogisticRegression(
    penalty='l1',
    sovler='liblinear'
)

tune_parameters = {
    'C':[0.001,0.01,0.1,1,10,100]
}
lr = GridSearchCV(lr_model_best,tune_parameters,scoring='roc_auc',cv=10)
lr.fit(X_train,y_train)
print(lr.best_params_)
print(lr.best_score_)