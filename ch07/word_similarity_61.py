from load_word_vector_60 import WordEmbedding

if __name__ == '__main__':
    filepath = './data/GoogleNews-vectors-negative300.bin'
    
    word_embedding = WordEmbedding()
    word_embedding.load_dataset(filepath)

    similarity = word_embedding.model.similarity('United_States', 'U.S.')
    print(similarity)
    '''
    0.73107743
    '''