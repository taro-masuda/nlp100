import pandas as pd
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from tqdm import tqdm

class WordEmbedding:
    def __init__(self):
        pass

    def load_dataset(self, filepath: str) -> None:
        self.model = KeyedVectors.load_word2vec_format(filepath,
                                                    binary=True)

    def word_vector(self, word: str) -> np.array:
        pass

    def analogous_vector(self, w1: str, w2: str, w3: str) -> np.array:
        if not w1 in self.model or not w2 in self.model or w3 not in self.model:
            return None
        else:
            return self.model[w2] - self.model[w1] + self.model[w3]

if __name__ == '__main__':
    filepath_word_embedding = './data/GoogleNews-vectors-negative300.bin'
    filepath_analogy_in = './data/questions-words.txt'
    
    word_embedding = WordEmbedding()
    word_embedding.load_dataset(filepath_word_embedding)
    model = word_embedding.model

    df_analogy = pd.read_table(filepath_analogy_in, 
                                sep='\s', 
                                skiprows=1,
                                header=None)
    df_analogy.columns = ['word1', 'word2', 'word3', 'word4']
    df_analogy['analogous_word'] = None
    df_analogy['similarity'] = None

    most_similar_words = []
    similarities = []
    for w1, w2, w3 in tqdm(zip(df_analogy['word1'],
                          df_analogy['word2'],
                          df_analogy['word3'])):
        if ':' in w1: 
            most_similar_words.append('')
            similarities.append('')
            continue
        most_similar_word, similarity = model.most_similar(
            positive=[w2, w3], negative=[w1], topn=1)[0]
        most_similar_words.append(most_similar_word)
        similarities.append(similarity)

    df_analogy['analogous_word'] = most_similar_words
    df_analogy['similarity'] = similarities

    print(df_analogy.head())

    filepath_analogy_out = filepath_analogy_in.replace('.txt', '_out.txt')
    df_analogy.to_csv(filepath_analogy_out, sep=' ',
                    index=None)
    
    '''

    '''
    
