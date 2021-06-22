import os
import matplotlib.pyplot as plt
import numpy as np

from reading_the_result_30 import load_morpho_dict_list
from word_frequency_35 import freq_count

if __name__ == '__main__':
    dirpath = './data'
    resultpath = os.path.join(dirpath, 'neko.txt.mecab')
    figpath = os.path.join(dirpath, 'word_frequency.png')

    doc_dic_list = load_morpho_dict_list(resultpath=resultpath)
    
    freq_list = freq_count(doc_dic_list)
    left = np.arange(10)

    height = [freq for (freq, _) in freq_list[:10]]
    labels = [word for (_, word) in freq_list[:10]]

    plt.bar(left, height, tick_label=labels)
    plt.savefig(figpath)