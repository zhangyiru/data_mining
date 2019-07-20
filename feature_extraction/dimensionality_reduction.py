from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import FastICA
import pandas as pd

"""简单设定了降维后的维度"""
def pca(data):
    pca = PCA(n_components=150)
    data = pca.fit_transform(data)
    return pd.DataFrame(data)


def lda(data,target):
    lda = LinearDiscriminantAnalysis(n_components=150)
    data = lda.fit_transform(data,target)
    return pd.DataFrame(data)

def lca(data):
    lca = FastICA(n_components=150)
    data = lca.fit_transform(data)
    return pd.DataFrame(data)
