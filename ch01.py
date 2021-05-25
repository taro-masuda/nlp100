

def reverse_string(string: str) -> str:
    return string[::-1]

def concat_odd_chars(string: str) -> str:
    return string[0::2]

def concat_each_char_1by1(str1: str, str2: str) -> str:
    output_str = ''
    for char1, char2 in zip(str1, str2):
        output_str += char1 + char2
    return output_str

def create_wordsize_list(text: str) -> list:
    word_list = text.replace(',', '').replace('.', '').split(' ')
    wordsize_list = []
    for word in word_list:
        wordsize_list.append(len(word))
    return wordsize_list

if __name__ == '__main__':
    print(reverse_string('stressed'))
    print(concat_odd_chars('パタトクカシーー'))
    print(concat_each_char_1by1('パトカー', 'タクシー'))
    
    text = 'Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics.'
    print(create_wordsize_list(text))