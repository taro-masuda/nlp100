import os
from reading_the_result_30 import load_morpho_dict_list
import numpy as np
import matplotlib.pyplot as plt
from word_frequency_35 import freq_count


if __name__ == '__main__':
    dirpath = './data'
    resultpath = os.path.join(dirpath, 'neko.txt.mecab')
    figpath = os.path.join(dirpath, 'zipf_law_39.png')

    doc_dic_list = load_morpho_dict_list(resultpath=resultpath)
    
    freq_list = freq_count(doc_dic_list)
    
    ax = plt.gca()
    ax.set_yscale('log')
    ax.set_xscale('log')

    for i, freqword in enumerate(freq_list):
        rank = i + 1
        freq = freqword[0]
        plt.scatter(rank, freq)
    
    plt.savefig(figpath)