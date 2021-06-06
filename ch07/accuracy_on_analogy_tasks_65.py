import pandas as pd
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    filepath = './data/questions-words_out.txt'

    df = pd.read_csv(filepath, sep=' ')
    drop_index = df.index[df['word1'].str.contains(':')]
    df = df.drop(drop_index)

    y_true = df['word4']
    y_pred = df['analogous_word']
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    print('Accuracy: {:.4f}'.format(acc))
    '''
    Accuracy: 0.7359
    '''