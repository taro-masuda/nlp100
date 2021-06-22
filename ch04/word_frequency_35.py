import os
from reading_the_result_30 import load_morpho_dict_list

def freq_count(doc_dic_list: list) -> list:
    freq_dic = {}
    for sentence_dic_list in doc_dic_list:
        for word_dic in sentence_dic_list:
            word = word_dic['base']
            freq_dic[word] = freq_dic.get(word, 0) + 1

    freq_list = [(freq, word) for word, freq in freq_dic.items()]
    freq_list = sorted(freq_list, reverse=True)
    return freq_list

if __name__ == '__main__':
    dirpath = './data'
    resultpath = os.path.join(dirpath, 'neko.txt.mecab')

    doc_dic_list = load_morpho_dict_list(resultpath=resultpath)
    
    freq_list = freq_count(doc_dic_list)
    

    for (freq, word) in freq_list[:20]:
        print(freq, word)
    for (freq, word) in freq_list[-20:]:
        print(freq, word)
    '''
    9194 の
    7486 。
    6853 て
    6772 、
    6422 は
    6268 に
    6071 を
    5978 だ
    5515 と
    5339 が
    4270 た
    3669 する
    3231 「
    3225 」
    3054 ない
    2479 も
    2322 ある
    2191 *
    2090 で
    2042 から
    1 あっけない
    1 あたし
    1 あたう
    1 あそこ
    1 あさって
    1 あさい
    1 あご
    1 あこがれる
    1 あげく
    1 あけ
    1 あぐり
    1 あくる
    1 あかんべえ
    1 あからさま
    1 あか
    1 あいだ
    1 あい
    1 』
    1 『
    1 〇
    '''

