import os
from reading_the_result_30 import load_morpho_dict_list

if __name__ == '__main__':
    dirpath = './data'
    resultpath = os.path.join(dirpath, 'neko.txt.mecab')

    doc_dic_list = load_morpho_dict_list(resultpath=resultpath)
    
    verbs = []
    for sentence_dic_list in doc_dic_list:
        for idx in range(len(sentence_dic_list)-2):
            dic1 = sentence_dic_list[idx]
            dic2 = sentence_dic_list[idx+1]
            dic3 = sentence_dic_list[idx+2]
            if dic1['pos'] == '名詞' and dic3['pos'] == '名詞' and dic2['surface'] == 'の':
                print(dic1['surface'], dic2['surface'], dic3['surface'])
                
    '''
    彼 の 掌
    掌 の 上
    書生 の 顔
    はず の 顔
    顔 の 真中
    ...
    年 の 間
    自然 の 力
    水 の 中
    座敷 の 上
    不可思議 の 太平
    '''