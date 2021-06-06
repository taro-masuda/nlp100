from sklearn.cluster import KMeans
import os
import random
import numpy as np
import pycountry

from load_word_vector_60 import WordEmbedding


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

if __name__ == "__main__":
    seed_everything()

    fp_wv = './data/GoogleNews-vectors-negative300.bin'
    word_embedding = WordEmbedding()
    word_embedding.load_dataset(fp_wv)

    country_set = set()
    for country in pycountry.countries:
        country_set.add(country.name)
        country_set.add(country.name.replace(' ', '_'))

    ndim = word_embedding.model["United_States"].shape[0]
    X = np.empty(shape=(1, ndim))
    for word in country_set:
        if word in word_embedding.model:
            vector = word_embedding.model[word]
            X = np.concatenate([X, vector.reshape(-1, ndim)], axis=0)

    kmeans = KMeans(n_clusters=5, random_state=42).fit(X)

    print(kmeans.cluster_centers_)
    '''
    [[ 0.10704973 -0.26239014  0.02558051 ... -0.01858605  0.18135749
    0.18623861]
    [-0.03105244 -0.00173609  0.01469943 ...  0.08371082  0.04689668
    0.18053557]
    [ 0.07657771 -0.08527736  0.07447788 ...  0.04669598  0.15532885
    0.03362556]
    [ 0.04348138  0.0967682   0.03860149 ... -0.06409113 -0.01980943
    0.06078452]
    [-0.09481733 -0.14094152  0.05045573 ... -0.0737649   0.21163901
    0.03291927]]
    '''