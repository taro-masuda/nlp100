from load_word_vector_60 import WordEmbedding

if __name__ == '__main__':
    filepath = './data/GoogleNews-vectors-negative300.bin'
    
    word_embedding = WordEmbedding()
    word_embedding.load_dataset(filepath)
    model = word_embedding.model

    result = model.similar_by_word("United_States")
    for rank in range(10):
        similar_key, similarity = result[rank] 
        print(f"{similar_key}: {similarity:.4f}")
    '''
    Unites_States: 0.7877
    Untied_States: 0.7541
    United_Sates: 0.7401
    U.S.: 0.7311
    theUnited_States: 0.6404
    America: 0.6178
    UnitedStates: 0.6167
    Europe: 0.6133
    countries: 0.6045
    Canada: 0.6019
    '''