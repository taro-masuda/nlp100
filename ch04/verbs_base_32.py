import os
from reading_the_result_30 import load_morpho_dict_list

if __name__ == '__main__':
    dirpath = './data'
    resultpath = os.path.join(dirpath, 'neko.txt.mecab')

    doc_dic_list = load_morpho_dict_list(resultpath=resultpath)
    
    verbs = []
    for sentence_dic_list in doc_dic_list:
        for dic in sentence_dic_list:
            if dic['pos'] == '動詞':
                verbs.append(dic['base'])
    print(verbs)
    '''
    ['生れる', 'つく', 'する', '泣く', 'する',
    ...
    '死ぬ', '死ぬ', '得る', '死ぬ', '得る', 'られる']
    '''