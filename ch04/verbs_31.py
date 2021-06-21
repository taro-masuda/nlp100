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
                verbs.append(dic['surface'])
    print(verbs)
    '''
    ['生れ', 'つか', 'し', '泣い', 'し', 'いる', '始め', '見', '聞く', '捕え', '煮', '食う', 
    ...
    '得る', '死な', '得', 'られ']
    '''