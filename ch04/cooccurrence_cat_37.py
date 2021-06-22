import os
from reading_the_result_30 import load_morpho_dict_list
import numpy as np
import matplotlib.pyplot as plt

def cooccurrence_count(doc_dic_list: list, keyword: str) -> list:
    freq_dic = {}
    for sentence_dic_list in doc_dic_list:
        word_list = [word_dic['base'] for word_dic in sentence_dic_list]
        word_set = set(word_list)
        if keyword in word_set:
            for word_dic in sentence_dic_list:
                word = word_dic['base']
                if not word == keyword:
                    freq_dic[word] = freq_dic.get(word, 0) + 1

    freq_list = [(freq, word) for word, freq in freq_dic.items()]
    freq_list = sorted(freq_list, reverse=True)
    return freq_list

if __name__ == '__main__':
    dirpath = './data'
    resultpath = os.path.join(dirpath, 'neko.txt.mecab')
    figpath = os.path.join(dirpath, 'cooccurrence_freq_neko.png')

    doc_dic_list = load_morpho_dict_list(resultpath=resultpath)
    
    freq_list = cooccurrence_count(doc_dic_list, keyword='猫')
    
    left = np.arange(10)

    height = [freq for (freq, _) in freq_list[:10]]
    labels = [word for (_, word) in freq_list[:10]]

    plt.bar(left, height, tick_label=labels)
    plt.savefig(figpath)
    '''

    '''

