import os
from reading_the_result_30 import load_morpho_dict_list
import numpy as np
import matplotlib.pyplot as plt
from word_frequency_35 import freq_count


if __name__ == '__main__':
    dirpath = './data'
    resultpath = os.path.join(dirpath, 'neko.txt.mecab')
    figpath = os.path.join(dirpath, 'histogram_38.png')

    doc_dic_list = load_morpho_dict_list(resultpath=resultpath)
    
    freq_list = freq_count(doc_dic_list)

    height = [freq for (freq, _) in freq_list]
    max_height = max(height)
    
    plt.hist(x=height, bins=100)#np.arange(1, max_height+1))
    plt.savefig(figpath)