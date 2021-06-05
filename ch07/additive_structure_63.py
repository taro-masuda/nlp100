from load_word_vector_60 import WordEmbedding

if __name__ == '__main__':
    filepath = './data/GoogleNews-vectors-negative300.bin'
    
    word_embedding = WordEmbedding()
    word_embedding.load_dataset(filepath)
    model = word_embedding.model

    result = model.most_similar(positive=['Spain', 'Athens'], negative=['Madrid'])
    for rank in range(10):
        similar_key, similarity = result[rank] 
        print(f"{similar_key}: {similarity:.4f}")
    '''
    Greece: 0.6898
    Aristeidis_Grigoriadis: 0.5607
    Ioannis_Drymonakos: 0.5553
    Greeks: 0.5451
    Ioannis_Christou: 0.5401
    Hrysopiyi_Devetzi: 0.5248
    Heraklio: 0.5208
    Athens_Greece: 0.5169
    Lithuania: 0.5167
    Iraklion: 0.5147
    '''