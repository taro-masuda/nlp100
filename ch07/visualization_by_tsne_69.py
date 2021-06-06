from sklearn.cluster import KMeans
import os
import random
import numpy as np
import pycountry
import matplotlib.pyplot as plt
from typing import Union
from sklearn.manifold import TSNE


from load_word_vector_60 import WordEmbedding

def load_country_set() -> set:
    country_set = set()
    for country in pycountry.countries:
        country_set.add(country.name)
        country_set.add(country.name.replace(' ', '_'))
    return country_set

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def load_country_vectors(country_set: set, 
                        word_vectors) -> Union[np.array, list]:
    ndim = word_vectors["United_States"].shape[0]
    X = np.empty(shape=(0, ndim))
    labels = []
    for word in country_set:
        if word in word_vectors:
            vector = word_vectors[word]
            labels.append(word)
            X = np.concatenate([X, vector.reshape(-1, ndim)], axis=0)
    return X, labels

if __name__ == "__main__":
    seed_everything()
    fp_wv = './data/GoogleNews-vectors-negative300.bin'
    fig_path = './data/t-sne_visualization.png'

    word_embedding = WordEmbedding()
    word_embedding.load_dataset(fp_wv)

    country_set = load_country_set()

    X, labels = load_country_vectors(country_set=country_set, word_vectors=word_embedding.model)
    X_embedded = TSNE(n_components=2).fit_transform(X)
    
    x = []
    y = []
    for i_row in range(X.shape[0]):
        x.append(X_embedded[i_row, 0])
        y.append(X_embedded[i_row, 1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                    xy=(x[i], y[i]))
    plt.savefig(fig_path)
    '''
    
    '''