

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

def generate_template_text(x: int, y: str, z: float) -> str:
    return '{0}時の{1}は{2}'.format(x,y,z)

def cipher(string: str) -> str:
    crypted_str = ''
    for i, char in enumerate(string):
        ascii_code = ord(char)
        if ascii_code >= 97 and ascii_code <= 122:
            crypted_str += chr(219 - ascii_code)
        else:
            crypted_str += char
    return crypted_str

if __name__ == '__main__':
    assert(reverse_string('stressed') == 'desserts')
    assert(concat_odd_chars('パタトクカシーー') == 'パトカー')
    assert(concat_each_char_1by1('パトカー', 'タクシー') == 'パタトクカシーー')
    
    text = 'Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics.'
    assert(create_wordsize_list(text) == [3,1,4,1,5,9,2,6,5,3,5,8,9,7,9])

    text = 'Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can.'
    assert(create_map_from_str_to_pos(text) == {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10, 'Na': 11, 'Mi': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20})

    assert(generate_template_text(12, '気温', 22.4) == '12時の気温は22.4')

    assert(cipher('abc!') == 'zyx!')
    assert(cipher(cipher('helloworld')) == 'helloworld')
    assert(cipher('!?#') == '!?#')

    print('All test cases passed.')