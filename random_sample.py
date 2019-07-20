import pandas as pd
from sklearn.model_selection import train_test_split
from model_code import lr,lasso,ridge,svm,gbdt,rf,xgb,LightGbm as lgb

def rand_sample(pos_sample,neg_sample,number):
  #采样比例
  neg_rate = 0.06
  auc = 0
  
  for i in range(number):

    
  
