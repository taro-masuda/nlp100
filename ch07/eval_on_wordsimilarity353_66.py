import os
from numpy.lib.npyio import save, savez_compressed
import pandas as pd
import requests
from sklearn.metrics import accuracy_score
import zipfile, io
from scipy.stats import spearmanr

from load_word_vector_60 import WordEmbedding

    
def fetch_file(url: str, savedir: str) -> str:
    savepath = os.path.join(savedir, url.split('/')[-1])
    req = requests.get(url, stream=True)
    zf = zipfile.ZipFile(io.BytesIO(req.content))
    zf.extractall(savedir)
    saved_dir = savepath.replace('.zip', '')
    #print(saved_dir)
    return saved_dir

if __name__ == "__main__":
    savedir = './data/'
    url = 'http://www.gabrilovich.com/resources/data/wordsim353/wordsim353.zip'
    zipdir = fetch_file(url=url, savedir=savedir)
    
    fp_human = os.path.join(zipdir, 'combined.csv')
    df = pd.read_csv(fp_human)
    df = df.sort_values(by=['Human (mean)'], ascending=False)

    fp_wv = './data/GoogleNews-vectors-negative300.bin'
    word_embedding = WordEmbedding()
    word_embedding.load_dataset(fp_wv)

    df['sim_word_vec'] = None
    df['sim_word_vec'] = [word_embedding.model.similarity(w1, w2)
                            for w1, w2 in zip(df['Word 1'], df['Word 2'])]
    coef1, pvalue = spearmanr(df['Human (mean)'], df['sim_word_vec'])
    print('Spearman Corr Coef: {:.4f}'.format(coef1))
    '''
    Spearman Corr Coef: 0.7000
    '''