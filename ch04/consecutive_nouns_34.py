import os
from reading_the_result_30 import load_morpho_dict_list

if __name__ == '__main__':
    dirpath = './data'
    resultpath = os.path.join(dirpath, 'neko.txt.mecab')

    doc_dic_list = load_morpho_dict_list(resultpath=resultpath)
    
    max_cons_nouns = []
    for sentence_dic_list in doc_dic_list:
        cons_nouns = []
        for word_dic in sentence_dic_list:
            if word_dic['pos'] == '名詞':
                cons_nouns.append(word_dic['surface'])
                if len(cons_nouns) > len(max_cons_nouns):
                    max_cons_nouns = cons_nouns
            else:
                cons_nouns = []
    print(max_cons_nouns)
    '''
    ['many', 'a', 'slip', "'", 'twixt', 'the', 'cup', 'and', 'the', 'lip']
    '''