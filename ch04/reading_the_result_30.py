from posixpath import split
import MeCab
import os

def load_text(filepath: str) -> str:
    with open(filepath, 'r') as f:
        text = f.read()
    #text = ' '.join(text.splitlines())
    return text

def save_text(text: str, filepath: str) -> None:
    with open(filepath, 'w') as f:
        f.write(text)

def load_morpho_dict_list(resultpath: str) -> list:
    dict_list = []
    with open(resultpath, 'r') as f:
        lines = f.readlines()
        l = []
        for line in lines:
            dic = {}
            splitline = line.split('\t')
            if len(splitline) == 1: continue
            surface = splitline[0]; rests = splitline[1]
            dic['surface'] = surface
            rests = rests.split(',')
            dic['base'] = rests[-3]
            dic['pos'] = rests[1]
            dic['pos1'] = rests[2]
            l.append(dic)
            if dic['surface'] == '。':
                dict_list.append(l)
                l = []
    return dict_list


if __name__ == '__main__':
    dirpath = './data'
    filepath = os.path.join(dirpath, 'neko.txt')
    resultpath = os.path.join(dirpath, 'neko.txt.mecab')
    text = load_text(filepath=filepath)    

    mecab = MeCab.Tagger()
    result = mecab.parse(text)
    save_text(text=result, filepath=resultpath)

    dic_list = load_morpho_dict_list(resultpath=resultpath)
    print(dic_list)
    '''
    [[{'surface': '一', 'base': '一', 'pos': '数', 'pos1': '*'}, 
    {'surface': '\u3000', 'base': '\u3000', 'pos': '空白', 'pos1': '*'}, 
    {'surface': '吾輩', 'base': '吾輩', 'pos': '代名詞', 'pos1': '一般'}, 
    {'surface': 'は', 'base': 'は', 'pos': '係助詞', 'pos1': '*'}, 
    {'surface': '猫', 'base': '猫', 'pos': '一般', 'pos1': '*'}, 
    {'surface': 'で', 'base': 'だ', 'pos': '*', 'pos1': '*'}, 
    {'surface': 'ある', 'base': 'ある', 'pos': '*', 'pos1': '*'}, 
    {'surface': '。', 'base': '。', 'pos': '句点', 'pos1': '*'}], 
    [{'surface': '名前', 'base': '名前', 'pos': '一般', 'pos1': '*'}, 
    {'surface': 'は', 'base': 'は', 'pos': '係助詞', 'pos1': '*'}, 
    {'surface': 'まだ', 'base': 'まだ', 'pos': '助詞類接続', 'pos1': '*'}, 
    {'surface': '無い', 'base': '無い', 'pos': '自立', 'pos1': '*'}, 
    {'surface': '。', 'base': '。', 'pos': '句点', 'pos1': '*'}],
    ...
    ]
    '''