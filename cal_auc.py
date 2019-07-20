from random_sample import rand_sample

def auc(data,model_name=None,iterator_times=100):
  org_pos_sample = data.loc[data['hypertension']!=0]
  org_neg_sample = data.loc[data['hypertension']==0]
  
  auc = rand_sample(org_pos_sample,org_neg_sample,model=model_name,number=iterator_times)
  return auc
