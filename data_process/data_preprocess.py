import pandas as pd
from sklearn.preprocessing import StandardScaler

def del_id_get_hot(data):
  
  data = pd.get_dummies(data)
  print("one_hot后的维度",data.shape)
  
  return data

def del_id_ss(data):
  data = data.copy()
  label = data.pop('hypertension')
  
  #提取所有数值型特征
  num_df = pd.DataFrame()
  for col in data.columns:
    if data[col].dtype != 'object':
      num_df[col]=data.loc[:,col]
      del data[col]
  
  #对数值型特征进行标准归一化
  scaler = StandardScaler()
  scaler.fit(num_df)
  
  #返回值为array,要转成dataframe
  num_df_after_ss = scaler.transform(num_df)
  num_df_after_ss = pd.DataFrame(num_df_after_ss)
  
  #对所有离散值进行one_hot处理
  discrete_df_after_oh = pd.get_dummies(data)
  
  #结合数值型特征和离散值特征
  new_data = pd.concat([num_df_after_ss,discrete_df_after_oh,label],axis=1)
  return new_data
  
