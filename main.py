#!/usr/bin/env python
# -*- coding:utf-8 -*-
import pandas as pd
from data_process import data_preprocess
from random_sample import rand_sample
from cal_auc import auc

#读取数据集
data = pd.read_csv("testset.csv",sep=",")

"""特征处理"""
data = data_preprocess.del_id_get_hot(data)

"""特征选择"""

"""选择正负样本"""
org_pos_sample = data.loc[data['hypertension']!=0]
org_neg_sample = data.loc[data['hypertension']==0]

"""选择模型，随机采样n次,返回auc的平均值"""
auc = rand_sample(org_pos_sample,org_neg_sample,"lr",100)
print(auc)