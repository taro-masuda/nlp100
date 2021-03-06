import random

def reverse_string(string: str) -> str:
    return string[::-1]

def concat_odd_chars(string: str) -> str:
    return string[0::2]

def concat_each_char_1by1(str1: str, str2: str) -> str:
    output_str = ''
    for char1, char2 in zip(str1, str2):
        output_str += char1 + char2
    return output_str

def replace_symbols_and_return_list(text: str) -> list:
    return text.replace(',', '').replace('.', '').split(' ')

def create_wordsize_list(text: str) -> list:
    word_list = replace_symbols_and_return_list(text)
    wordsize_list = []
    for word in word_list:
        wordsize_list.append(len(word))
    return wordsize_list

def create_map_from_str_to_pos(text: str) -> dict:
    word_list = replace_symbols_and_return_list(text)
    mp = {}
    for idx, word in enumerate(word_list):
        if idx + 1 in (1,5,6,7,8,9,15,16,19):
            mp[word[0]] = idx + 1
        else:
            mp[word[0:2]] = idx + 1
    return mp

def generate_ngram(sequence, n: int) -> list:
    ngram = []
    for idx in range(len(sequence) - n + 1):
        ngram.append(sequence[idx:idx + n])
    return ngram

def generate_bigram_set(string: str) -> set:
    bigrams = generate_ngram(list(string), n=2)
    bigram_set = set()
    for bigram in bigrams:
        bigram_set.add(tuple(bigram))
    return bigram_set

def test_for_set_06() -> None:
    str1 = 'paraparaparadise'
    str2 = 'paragraph'
    X = generate_bigram_set(str1)
    print(X, type(X))
    # assert X == {('p', 'a'), ('a', 'r'), ('r', 'a'), ('a', 'p'), 
    #            ('a', 'd'), ('d', 'i'), ('i', 's'), ('s', 'e')}
    Y = generate_bigram_set(str2)
    print(Y)
    # assert Y == {('p', 'a'), ('a', 'r'), ('r', 'a'), ('a', 'p'), 
    #            ('a', 'g'), ('g', 'r'), ('p', 'h')}
    print(X | Y)
    # assert X | Y == {('p', 'a'), ('a', 'r'), ('r', 'a'), 
    #            ('a', 'p'), ('a', 'd'), ('d', 'i'), ('i', 's'), ('s', 'e'),
    #            ('a', 'g'), ('g', 'r'), ('p', 'h')}
    print(X & Y)
    # assert X & Y == {('p', 'a'), ('a', 'r'), ('r', 'a'), ('a', 'p')}
    assert X - Y == {('a', 'd'), ('d', 'i'), ('i', 's'), ('s', 'e')}
    assert Y - X == {('a', 'g'), ('g', 'r'), ('p', 'h')}
    assert ('s', 'e') in X
    assert ('s', 'e') not in Y

def generate_template_text(x: int, y: str, z: float) -> str:
    return '{0}??????{1}???{2}'.format(x,y,z)

def cipher(string: str) -> str:
    crypted_str = ''
    for i, char in enumerate(string):
        ascii_code = ord(char)
        if ascii_code >= 97 and ascii_code <= 122:
            crypted_str += chr(219 - ascii_code)
        else:
            crypted_str += char
    return crypted_str

def typoglycemia(text: str) -> str:
    word_list = text.split(' ')
    for i, word in enumerate(word_list):
        if len(word) > 4:
            char_list = list(word)[1:-1]
            random.shuffle(char_list)
            word_list[i] = word[0] + ''.join(char_list) + word[-1]
            
    return ' '.join(word_list)

if __name__ == '__main__':
    assert reverse_string('stressed') == 'desserts'
    assert concat_odd_chars('????????????????????????') == '????????????'
    assert concat_each_char_1by1('????????????', '????????????') == '????????????????????????'
    
    text = 'Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics.'
    assert create_wordsize_list(text) == [3,1,4,1,5,9,2,6,5,3,5,8,9,7,9]

    text = 'Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can.'
    assert create_map_from_str_to_pos(text) == {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10, 'Na': 11, 'Mi': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20}

    assert generate_ngram('I am an NLPer'.split(' '), n=2) == [['I', 'am'], ['am', 'an'], ['an', 'NLPer']]
    assert generate_ngram(list('I am an NLPer'.replace(' ', '')), n=2) == [['I', 'a'], ['a', 'm'], ['m', 'a'], ['a', 'n'], ['n', 'N'], ['N', 'L'], ['L', 'P'], ['P', 'e'], ['e', 'r']]

    test_for_set_06()

    assert generate_template_text(12, '??????', 22.4) == '12???????????????22.4'

    assert cipher('abc!') == 'zyx!'
    assert cipher(cipher('helloworld')) == 'helloworld'
    assert cipher('!?#') == '!?#'

    print(typoglycemia('I couldn???t believe that I could actually understand what I was reading : the phenomenal power of the human mind .'))
    print('All test cases passed.')